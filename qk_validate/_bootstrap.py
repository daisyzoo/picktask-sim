"""引导：只读复用 picktask/lib，不修改 picktask 代码。"""

from __future__ import annotations

import sys
from pathlib import Path

QK_VALIDATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = QK_VALIDATE_DIR.parent
PICKTASK_LIB = REPO_ROOT / "picktask" / "lib"


def bootstrap() -> Path:
    if not PICKTASK_LIB.is_dir():
        raise SystemExit(f"找不到 picktask/lib: {PICKTASK_LIB}")
    if str(PICKTASK_LIB) not in sys.path:
        sys.path.insert(0, str(PICKTASK_LIB))
    from paths import setup_lib_paths

    setup_lib_paths(include_lerobot=False)
    return REPO_ROOT
