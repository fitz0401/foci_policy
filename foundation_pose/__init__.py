"""Vendored FoundationPose integration used by FOCI Policy.

The upstream snapshot still uses imports such as ``learning.datasets`` and
``Utils``. Expose this package directory for those imports without relying on
an editable install from another workspace or on the process working directory.
"""

from pathlib import Path
import sys


_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
