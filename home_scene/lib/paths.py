"""home_scene 路径常量与 sys.path 引导。

禁止依赖 picktask / vla_sim_eval；仅只读引用仓库根下 unitree_ros。
"""

from __future__ import annotations

import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
HOME_SCENE_DIR = LIB_DIR.parent
WORKSPACE_DIR = HOME_SCENE_DIR.parent

ASSETS_DIR = HOME_SCENE_DIR / "assets"
CACHE_DIR = ASSETS_DIR / "cache"
FIXTURES_DIR = ASSETS_DIR / "fixtures"
OBJECTS_DIR = ASSETS_DIR / "objects"
ROOMS_DIR = ASSETS_DIR / "rooms"
ROBOCASA_DIR = ASSETS_DIR / "robocasa"

DATA_DIR = HOME_SCENE_DIR / "data"
SESSIONS_DATA_DIR = DATA_DIR / "sessions"

# 只读：官方 G1 描述（不复制进 home_scene，避免双份 mesh）
G1_DESCRIPTION_DIR = WORKSPACE_DIR / "unitree_ros" / "robots" / "g1_description"
G1_URDF_PATH = G1_DESCRIPTION_DIR / "g1_23dof_mode_10.urdf"
G1_COMPILED_MJCF_CACHE = CACHE_DIR / "g1_23dof_mode_10_compiled_cache.xml"


def setup_lib_paths() -> None:
    """将 home_scene/lib 加入 sys.path。"""
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))


def assert_decoupled() -> None:
    """运行时粗检：sys.modules 中不应已加载 picktask / vla_sim_eval 业务模块。"""
    banned_prefixes = (
        "pickcup",
        "liftbag",
        "liftbag_sim_core",
        "pickcup_sim_core",
        "lerobot_recorder",
    )
    leaked = [
        name
        for name in sys.modules
        if name in banned_prefixes or name.startswith("evaluation.")
    ]
    # 允许偶然同名第三方包；这里只拦我们已知的业务短名
    if leaked:
        raise RuntimeError(
            "home_scene 必须与 picktask/vla_sim_eval 解耦，"
            f"但检测到已加载模块: {sorted(leaked)}"
        )
