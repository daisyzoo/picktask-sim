#!/usr/bin/env python3
"""静态检查：源码不得 import picktask / vla_sim_eval 业务模块。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_MODULES = frozenset(
    {
        "pickcup",
        "liftbag",
        "liftbag_sim_core",
        "pickcup_sim_core",
        "lerobot_recorder",
        "picktask",
        "vla_sim_eval",
        "evaluation",
    }
)


def _imported_roots(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


def main() -> int:
    bad: list[str] = []
    for path in sorted((ROOT / "lib").rglob("*.py")):
        for name in _imported_roots(path):
            if name in BANNED_MODULES:
                bad.append(f"{path.relative_to(ROOT)}: import {name}")

    if bad:
        print("FAIL decoupling:")
        for line in bad:
            print(" ", line)
        return 1
    print("OK: no picktask/vla_sim_eval imports in home_scene/lib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
