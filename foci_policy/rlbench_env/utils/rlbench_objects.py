"""
This file contains a dictionary that maps task names to their corresponding grasp and target object names.
The dictionary is used to define the objects involved in each task for the RLBench environment.
"""
from rlbench.const import colors
from rlbench.tasks.put_groceries_in_cupboard import GROCERY_NAMES
from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES

dustpan_sizes = ['tall', 'short']

task_object_dict = {
    "meat_off_grill": {
        "grasp_object_name": {
            0: "chicken",
            1: "steak"
        },
        "target_object_name": "grill",
    },
    "turn_tap": {
        "grasp_object_name": {
            0: "tap_left",
            1: "tap_right"
        },
        "target_object_name": "tap_main",
    },
    "close_jar": {
        "grasp_object_name": "jar_lid0", #{i: f"jar_lid{i % 2}" for i in range(len(colors))},
        "target_object_name": {i: f"jar{i % 2}" for i in range(len(colors))},
    },
    "reach_and_drag": {
        "grasp_object_name": {
            0: "stick",
            1: "cube"
        },
        "target_object_name": "target0",
    },
    "stack_blocks": { # multiple stage
        "grasp_object_name": {i: f"stack_blocks_target{i}" for i in range(4)},
        "target_object_name": {
            0: "stack_blocks_target_plane",
            1: f"stack_blocks_target{0}",
            2: f"stack_blocks_target{1}",
            3: f"stack_blocks_target{2}",
        },
    },
    "light_bulb_in": {
        "grasp_object_name": {i: f"light_bulb{i % 2}" for i in range(len(colors))},
        "target_object_name": "lamp_base",
    },
    "put_money_in_safe": {
        "grasp_object_name": "dollar_stack",
        "target_object_name": "safe_body",
    },
    "place_wine_at_rack_location": {
        "grasp_object_name": "wine_bottle",
        "target_object_name": "rack_top",
    },
    "put_groceries_in_cupboard":{
        "grasp_object_name": {i: GROCERY_NAMES[i].replace(' ', '_') for i in range(len(GROCERY_NAMES))},
        "target_object_name": "cupboard",
    },
    "place_shape_in_shape_sorter":{
        "grasp_object_name": {i: SHAPE_NAMES[i].replace(' ', '_') for i in range(len(SHAPE_NAMES))},
        "target_object_name": "shape_sorter",
    },
    "insert_onto_square_peg":{
        "grasp_object_name": "square_ring",
        "target_object_name": {
            0: "pillar0",
            1: "pillar1",
            2: "pillar2",
        }
    },
    "stack_cups": { # multiple stage
        "grasp_object_name": {
            0: "cup1",
            1: "cup3",
        },
        "target_object_name": {
            0: "cup2",
            1: "cup1",
        },
    },
    "place_cups": { # multiple stage
        "grasp_object_name": {i: f"mug{i}" for i in range(3)},
        "target_object_name": {
            0: "place_cups_holder_base",
            1: "mug3",
        },
    },
    "put_item_in_drawer": { # multiple stage
        "grasp_object_name": {
            0: "drawer_bottom",
            1: "drawer_middle",
            2: "drawer_top",
            3: "item",
        },
        "target_object_name": "drawer_frame",
    },
    "sweep_to_dustpan_of_size": {
        "grasp_object_name": {
            0: "broom",
            1: "broom_holder",
        },
        "target_object_name": {
            0: "dustpan_tall",
            1: "dustpan_short",
        },
    },
    "push_buttons": {
        "grasp_object_name": "push_buttons_target0",
        "target_object_name": "push_buttons_target0",
    },
    "slide_block_to_color_target": {
        "grasp_object_name": "block",
        "target_object_name": {
            0: "target1",
            1: "target2",
            2: "target3",
            3: "target4",
        },
    },
    "open_drawer": {
        "grasp_object_name": {
            0: "drawer_bottom",
            1: "drawer_middle",
            2: "drawer_top",
        },
        "target_object_name": "drawer_frame",
    },
    "close_box": {
        "grasp_object_name": "box_lid",
        "target_object_name": "box_base",
    }
}
