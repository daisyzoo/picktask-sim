"""脚本入口共用的 sys.path 引导。"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PICKTASK_DIR = SCRIPTS_DIR.parent
LIB_DIR = PICKTASK_DIR / "lib"


def bootstrap(*, include_lerobot: bool = False) -> Path:
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))
    from paths import setup_lib_paths

    setup_lib_paths(include_lerobot=include_lerobot)
    return PICKTASK_DIR
