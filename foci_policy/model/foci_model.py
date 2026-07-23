"""
Two-stage trajectory prediction with GMM goal and action generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from foci_policy.model.point_mae_modules import MaskedPointNetEncoder


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding with sinusoidal embeddings"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        """t: (B,) or (B, T)"""
        if t.dim() == 1:
            return self.pe[t]  # (B, d_model)
        else:
            return self.pe[t]  # (B, T, d_model)


class PoseEncoder(nn.Module):
    """Encoder for robot pose (9D: xyz + 6D rotation)"""
    def __init__(self, pose_dim=9, hidden_dim=32, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pose):
        """pose: (B, 9) or (B, T, 9)"""
        return self.encoder(pose)


class ObjectCrossAttention(nn.Module):
    """Cross-attention module between two object feature sets"""
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        # pa → pb cross attention (pb attend to pa)
        self.pb_attend_pa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # pb → pa cross attention (pa attend to pb)
        self.pa_attend_pb = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # FFN layers
        self.pa_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.pb_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_pa = nn.LayerNorm(d_model)
        self.norm_pb = nn.LayerNorm(d_model)

    def forward(self, pa_feat, pb_feat):
        """
        pa_feat: (B, L_a, D) - base object features
        pb_feat: (B, L_b, D) - moving object features
        """
        # pb attends to pa (pb queries, pa as key/value)
        pb_cross, _ = self.pb_attend_pa(pb_feat, pa_feat, pa_feat)
        pb_feat = self.norm_pb(pb_feat + pb_cross)
        pb_feat = pb_feat + self.pb_ffn(pb_feat)
        # pa attends to pb (pa queries, pb as key/value)
        pa_cross, _ = self.pa_attend_pb(pa_feat, pb_feat, pb_feat)
        pa_feat = self.norm_pa(pa_feat + pa_cross)
        pa_feat = pa_feat + self.pa_ffn(pa_feat)
        return pa_feat, pb_feat


class GaussianMixtureMDN(nn.Module):
    """GMM-based goal predictor"""
    def __init__(self, input_dim=512, hidden_dim=256, num_gaussians=5, output_dim=9):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.pi_net = nn.Linear(hidden_dim, num_gaussians)
        self.mu_net = nn.Linear(hidden_dim, num_gaussians * output_dim)
        self.sigma_net = nn.Linear(hidden_dim, num_gaussians * output_dim)
    
    def forward(self, context):
        """
        Args:
            context: (B, L, D)
        Returns:
            pi: (B, K), mu: (B, K, 9), sigma: (B, K, 9)
        """
        pooled = context.mean(dim=1)
        features = self.feature_net(pooled)
        
        pi = F.softmax(self.pi_net(features), dim=-1)
        mu = self.mu_net(features).view(-1, self.num_gaussians, self.output_dim)
        sigma = F.softplus(self.sigma_net(features)).view(-1, self.num_gaussians, self.output_dim) + 1e-6
        
        return pi, mu, sigma
    
    def sample(self, pi, mu, sigma, deterministic=False):
        """Sample from GMM"""
        B = pi.shape[0]
        batch_idx = torch.arange(B, device=pi.device)
        
        if deterministic:
            max_indices = torch.argmax(pi, dim=1)
            samples = mu[batch_idx, max_indices]
        else:
            component_indices = torch.multinomial(pi, num_samples=1).squeeze(-1)
            selected_mu = mu[batch_idx, component_indices]
            selected_sigma = sigma[batch_idx, component_indices]
            noise = torch.randn_like(selected_mu)
            samples = selected_mu + noise * selected_sigma
        
        return samples


class WaypointDecoder(nn.Module):
    """Goal-conditioned waypoint decoder"""
    def __init__(self, d_model=512, num_waypoints=4, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_waypoints = num_waypoints
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(9, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.waypoint_queries = nn.Parameter(torch.randn(num_waypoints, d_model))
        self.time_pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=num_waypoints)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.waypoint_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 9),
        )
    
    def forward(self, context, goal_pose):
        """
        Args:
            context: (B, L, D)
            goal_pose: (B, 9)
        Returns:
            pred_waypoints: (B, num_waypoints, 9) - waypoints from start to goal (excluding goal)
        """
        B = context.shape[0]
        
        # 1. Encode Goal
        goal_feat = self.goal_encoder(goal_pose)  # (B, D)
        
        # 2. Construct Query: Inject Goal into each waypoint query
        tgt = self.waypoint_queries.unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        tgt = tgt + goal_feat.unsqueeze(1)  # [Inject Goal into Queries]
        
        # 3. Add temporal positional encoding
        time_indices = torch.arange(self.num_waypoints, device=context.device)
        time_pos = self.time_pos_encoder(time_indices)
        tgt = tgt + time_pos.unsqueeze(0)
        
        # 4. Decode
        decoder_out = self.decoder(tgt, context)
        pred_waypoints = self.waypoint_head(decoder_out)
        
        return pred_waypoints


class FOCIModel(nn.Module):
    """Focus On Context and Interaction Model"""
    def __init__(
        self,
        point_encoder_cfg: dict,
        pcd_dim: int = 256,
        pose_dim: int = 128,
        prediction_length: int = 5,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        use_language: bool = False,
        mode: str = 'manip',
        use_pcd_features: bool = True,
        use_pose_features: bool = True,
        num_gaussians: int = 5,
        goal_hidden_dim: int = 256,
        waypoint_nhead: int = 4,
        waypoint_num_layers: int = 2,
    ):
        super().__init__()
        
        self.pcd_dim = pcd_dim
        self.prediction_length = prediction_length
        self.use_language = use_language
        self.mode = mode
        self.use_pcd_features = use_pcd_features
        self.use_pose_features = use_pose_features
        
        if not use_pcd_features and not use_pose_features:
            raise ValueError("At least one of use_pcd_features or use_pose_features must be True")
        
        # Point cloud encoders
        if use_pcd_features:
            point_encoder_cfg['masked_encoder_cfg'].embed_dim = pcd_dim
            self.pa_encoder = MaskedPointNetEncoder(**point_encoder_cfg)
            self.pb_encoder = MaskedPointNetEncoder(**point_encoder_cfg)
        
        # Pose encoder
        if use_pose_features:
            self.pose_encoder = PoseEncoder(pose_dim=9, output_dim=pose_dim)
            self.pose_proj = nn.Linear(pose_dim, pcd_dim)
        
        # Object cross attention (manip mode only)
        if use_pcd_features and mode == 'manip':
            self.pcd_cross_attn = ObjectCrossAttention(d_model=pcd_dim, nhead=nhead, dropout=dropout)
        
        # Frame fusion
        self.frame_fusion = nn.TransformerEncoderLayer(
            d_model=pcd_dim, nhead=nhead, dim_feedforward=pcd_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        # Language projection
        if use_language:
            self.language_proj = nn.Linear(512, pcd_dim)
        
        # Stage 1: Goal predictor
        self.goal_predictor = GaussianMixtureMDN(
            input_dim=pcd_dim, hidden_dim=goal_hidden_dim,
            num_gaussians=num_gaussians, output_dim=9
        )
        
        # Stage 2: Waypoint decoder
        num_waypoints = prediction_length - 1
        self.waypoint_decoder = WaypointDecoder(
            d_model=pcd_dim, num_waypoints=num_waypoints,
            nhead=waypoint_nhead, num_layers=waypoint_num_layers, dropout=dropout
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _encode_context(self, pa_points, pb_points, pa_pose, pb_pose, language_embedding=None):
        """Encode context features"""
        tokens = []
        
        if self.use_pcd_features:
            pa_feat, _ = self.pa_encoder(pa_points)
            pb_feat, _ = self.pb_encoder(pb_points)
            if self.mode == 'manip':
                pa_feat, pb_feat = self.pcd_cross_attn(pa_feat, pb_feat)
            tokens.extend([pa_feat, pb_feat])
        
        if self.use_pose_features:
            pa_pose_feat = self.pose_proj(self.pose_encoder(pa_pose)).unsqueeze(1)
            pb_pose_feat = self.pose_proj(self.pose_encoder(pb_pose)).unsqueeze(1)
            tokens.extend([pa_pose_feat, pb_pose_feat])
        
        if self.use_language and language_embedding is not None:
            lang_feat = self.language_proj(language_embedding).unsqueeze(1)
            tokens.append(lang_feat)
        
        frame_tokens = torch.cat(tokens, dim=1)
        fused_context = self.frame_fusion(frame_tokens)
        
        return fused_context
    
    def forward(self, pa_points, pb_points, pa_pose, pb_pose, language_embedding=None,
                return_gmm_params=False, deterministic_goal=False, gt_goal=None):
        fused_context = self._encode_context(pa_points, pb_points, pa_pose, pb_pose, language_embedding)

        # Stage 1: Predict goal
        pi, mu, sigma = self.goal_predictor(fused_context)
        goal_pose = self.goal_predictor.sample(pi, mu, sigma, deterministic=deterministic_goal)

        # Stage 2: Predict waypoints (excluding goal)
        # Teacher forcing: use GT goal during training so waypoint decoder learns correct conditioning
        waypoint_goal = gt_goal if gt_goal is not None else goal_pose
        pred_waypoints = self.waypoint_decoder(fused_context, waypoint_goal)

        # Concatenate waypoints with goal to form complete trajectory
        pred_trajectory = torch.cat([pred_waypoints, goal_pose.unsqueeze(1)], dim=1)

        if return_gmm_params:
            return goal_pose, pred_waypoints, pred_trajectory, (pi, mu, sigma)
        else:
            return pred_trajectory
    
    @torch.no_grad()
    def inference(self, pa_points, pb_points, pa_pose, pb_pose, language_embedding=None, deterministic=True):
        """Inference mode"""
        self.eval()
        return self.forward(pa_points, pb_points, pa_pose, pb_pose, language_embedding,
                            return_gmm_params=False, deterministic_goal=deterministic)


class FOCILoss(nn.Module):
    """Loss function for FOCI model"""
    def __init__(
        self,
        lambda_nll: float = 1.0,
        lambda_waypoint_pos: float = 1.0,
        lambda_waypoint_rot: float = 5.0,
        lambda_stage1: float = 0.3,
        lambda_stage2: float = 0.7,
        gamma_geo: float = 0.1,
    ):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_waypoint_pos = lambda_waypoint_pos
        self.lambda_waypoint_rot = lambda_waypoint_rot
        self.lambda_stage1 = lambda_stage1
        self.lambda_stage2 = lambda_stage2
        self.gamma_geo = gamma_geo
    
    def _gram_schmidt(self, a1, a2):
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)
    
    def gmm_nll_loss(self, pi, mu, sigma, target):
        """GMM negative log-likelihood"""
        B, K, D = mu.shape
        target = target.unsqueeze(1).expand(-1, K, -1)
        
        diff = target - mu
        log_prob = -0.5 * torch.sum((diff / sigma) ** 2, dim=-1)
        log_prob = log_prob - torch.sum(torch.log(sigma), dim=-1)
        log_prob = log_prob - 0.5 * D * math.log(2 * math.pi)
        
        log_pi = torch.log(pi + 1e-8)
        log_mixture = torch.logsumexp(log_pi + log_prob, dim=1)
        
        return -log_mixture.mean()
    
    def waypoint_loss(self, pred_trajectory, gt_trajectory):
        """Waypoint trajectory prediction loss"""
        pred_pos = pred_trajectory[:, :, :3]
        pred_rot_6d = pred_trajectory[:, :, 3:]
        gt_pos = gt_trajectory[:, :, :3]
        gt_rot_6d = gt_trajectory[:, :, 3:]
        
        pos_loss = F.smooth_l1_loss(pred_pos, gt_pos)
        rot_recon_loss = F.smooth_l1_loss(pred_rot_6d, gt_rot_6d)
        
        pred_R = self._gram_schmidt(pred_rot_6d[:, :, :3], pred_rot_6d[:, :, 3:])
        gt_R = self._gram_schmidt(gt_rot_6d[:, :, :3], gt_rot_6d[:, :, 3:])
        rot_geo_loss = F.l1_loss(pred_R, gt_R)
        
        rot_loss = rot_recon_loss + self.gamma_geo * rot_geo_loss
        
        return self.lambda_waypoint_pos * pos_loss + self.lambda_waypoint_rot * rot_loss, {
            'waypoint_pos': pos_loss,
            'waypoint_rot': rot_loss,
        }
    
    def forward(self, phase, goal_pose, pred_waypoints, pred_trajectory, gmm_params, gt_trajectory):
        """
        Compute loss based on training phase (3-stage).
        phase: 'stage1', 'stage2', or 'stage3'
        """
        losses = {}

        if phase == 'stage1':
            # Encoder + GMM only
            pi, mu, sigma = gmm_params
            gt_goal = gt_trajectory[:, -1, :]
            nll = self.gmm_nll_loss(pi, mu, sigma, gt_goal)
            losses['nll'] = nll
            total_loss = self.lambda_nll * nll

        elif phase == 'stage2':
            # Waypoint decoder with GT goal (teacher forcing) — no GMM loss, encoder frozen
            waypoint_loss, waypoint_losses = self.waypoint_loss(pred_waypoints, gt_trajectory[:, :-1, :])
            losses.update(waypoint_losses)
            total_loss = waypoint_loss

        elif phase == 'stage3':
            # Joint training with predicted goal (no teacher forcing)
            pi, mu, sigma = gmm_params
            gt_goal = gt_trajectory[:, -1, :]
            nll = self.gmm_nll_loss(pi, mu, sigma, gt_goal)
            waypoint_loss, waypoint_losses = self.waypoint_loss(pred_waypoints, gt_trajectory[:, :-1, :])
            losses['nll'] = nll
            losses.update(waypoint_losses)
            total_loss = self.lambda_stage1 * nll + self.lambda_stage2 * waypoint_loss

        else:
            raise ValueError(f"Unknown phase: {phase}")

        losses['total'] = total_loss
        return total_loss, losses
