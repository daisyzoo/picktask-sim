#!/usr/bin/env python3
"""将 RoboCasa 厨房资产接入 home_scene/assets/robocasa。

用法（任选其一）:
  # 1) 已安装 robocasa：软链其 models/assets
  python scripts/fetch_robocasa_assets.py --link

  # 2) 调用 robocasa 官方下载（约数 GB；需已 pip install robocasa）
  python scripts/fetch_robocasa_assets.py --download --type fixtures_lw objs_lw tex

  # 3) 从环境变量路径复制/登记
  #    export HOME_SCENE_ROBOCASA_ROOT=/path/to/robocasa/models/assets
  python scripts/fetch_robocasa_assets.py --from-env

  # 4) 仅扫描已有目录并写 manifest
  python scripts/fetch_robocasa_assets.py --scan
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from asset_registry import default_registry  # noqa: E402
from paths import ROBOCASA_DIR  # noqa: E402
from robocasa_bridge import (  # noqa: E402
    DEFAULT_FETCH_TYPES,
    ensure_dirs,
    env_override_root,
    link_from_robocasa_package,
    link_from_root,
    scan_and_register,
    status,
)


def _download_via_robocasa(types: list[str]) -> int:
    try:
        import robocasa  # noqa: F401
    except ImportError:
        print(
            "未安装 robocasa。请先:\n"
            "  pip install robocasa\n"
            "或改用 --link / --from-env",
            file=sys.stderr,
        )
        return 1
    # 官方脚本会下载到 robocasa 包内；下载后再 link 到本工程
    cmd = [
        sys.executable,
        "-m",
        "robocasa.scripts.download_kitchen_assets",
        "--type",
        *types,
    ]
    print("运行:", " ".join(cmd))
    print("注意: 官方脚本可能交互确认下载体积。")
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    linked = link_from_robocasa_package()
    print(f"已链接到 {linked}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link", action="store_true", help="软链已安装 robocasa 的 assets")
    parser.add_argument("--download", action="store_true", help="调用官方下载脚本")
    parser.add_argument(
        "--type",
        nargs="+",
        default=list(DEFAULT_FETCH_TYPES),
        help=f"下载类型，默认 {DEFAULT_FETCH_TYPES}",
    )
    parser.add_argument("--from-env", action="store_true", help="从 HOME_SCENE_ROBOCASA_ROOT 链接")
    parser.add_argument("--scan", action="store_true", help="扫描并登记现有 XML")
    args = parser.parse_args()

    ensure_dirs()
    did = False

    if args.from_env:
        src = env_override_root()
        if src is None:
            print("未设置有效的 HOME_SCENE_ROBOCASA_ROOT / ROBOCASA_ASSETS", file=sys.stderr)
            return 1
        link_from_root(src)
        print(f"已链接子目录 <- {src}")
        did = True

    if args.link:
        linked = link_from_robocasa_package()
        if linked is None:
            print(
                "无法定位 robocasa models/assets。\n"
                "可用:\n"
                "  export HOME_SCENE_ROBOCASA_ROOT="
                "/Users/ke/Documents/wm/mujoco/robocasa_ws/robocasa/robocasa/models/assets\n"
                "  python scripts/fetch_robocasa_assets.py --from-env --scan",
                file=sys.stderr,
            )
            return 1
        print(f"已链接 robocasa assets -> {linked}")
        did = True

    if args.download:
        rc = _download_via_robocasa(args.type)
        if rc != 0:
            return rc
        did = True

    if args.scan or did or not any([args.link, args.download, args.from_env, args.scan]):
        # 默认也 scan
        reg = default_registry(scan_robocasa=False)
        added = scan_and_register(reg)
        print(f"登记 RoboCasa 资产 {len(added)} 个 -> {ROBOCASA_DIR}")

    st = status()
    print(f"状态: available={st.available} n_xml={st.n_xml} ({st.note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
