"""picktask 路径常量与 sys.path 引导。"""

from __future__ import annotations

import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
PICKTASK_DIR = LIB_DIR.parent
WORKSPACE_DIR = PICKTASK_DIR.parent

ASSETS_DIR = PICKTASK_DIR / "assets"
CACHE_DIR = ASSETS_DIR / "cache"
IMG_DIR = ASSETS_DIR / "img"
YELLOWCUP_ASSETS_DIR = ASSETS_DIR / "yellowcupassets"
LIFTBAG_ASSETS_DIR = PICKTASK_DIR / "liftbagassets"

G1_DESCRIPTION_DIR = WORKSPACE_DIR / "unitree_ros" / "robots" / "g1_description"
G1_URDF_PATH = G1_DESCRIPTION_DIR / "g1_23dof_mode_10.urdf"
G1_COMPILED_MJCF_CACHE = CACHE_DIR / "g1_23dof_mode_10_compiled_cache.xml"

DATA_DIR = PICKTASK_DIR / "pickcupdata"
TRAIN_DATA_DIR = DATA_DIR / "train"
SESSIONS_DATA_DIR = DATA_DIR / "sessions"
ARCHIVE_DATA_DIR = DATA_DIR / "archive"

LIFTBAG_DATA_DIR = PICKTASK_DIR / "liftbagdata"
LIFTBAG_SESSIONS_DATA_DIR = LIFTBAG_DATA_DIR / "sessions"

LEROBOT_SRC = WORKSPACE_DIR / "lerobot" / "src"


def setup_lib_paths(*, include_lerobot: bool = False) -> None:
    """将 lib/（及可选 lerobot/src）加入 sys.path。"""
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))
    if include_lerobot and LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
