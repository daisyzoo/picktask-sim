"""RoboCasa 资产桥接：扫描 / 登记 / 可选下载到 home_scene/assets/robocasa。

不强制完整安装 robocasa 训练栈；未下载时场景继续用 stub。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from asset_registry import AssetRegistry, AssetSpec
from paths import ASSETS_DIR

ROBOCASA_DIR = ASSETS_DIR / "robocasa"
ROBOCASA_FIXTURES_DIR = ROBOCASA_DIR / "fixtures"
ROBOCASA_OBJECTS_DIR = ROBOCASA_DIR / "objects"
ROBOCASA_TEXTURES_DIR = ROBOCASA_DIR / "textures"
MANIFEST_PATH = ROBOCASA_DIR / "manifest.json"
DEFAULT_WS_ASSETS = (
    ASSETS_DIR.parent.parent / "robocasa_ws" / "robocasa" / "robocasa" / "models" / "assets"
)

# 轻量包：足够后续换柜门/小物体；完整包约 10GB+
DEFAULT_FETCH_TYPES = ("fixtures_lw", "objs_lw", "tex")


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def iter_asset_files(root: Path) -> list[Path]:
    """递归列出 xml/mjcf；显式 followlinks（macOS Path.rglob 常不跟目录软链）。"""
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if Path(name).suffix.lower() in {".xml", ".mjcf"}:
                out.append(Path(dirpath) / name)
    return out


@dataclass(frozen=True)
class RoboCasaStatus:
    root: Path
    available: bool
    n_xml: int
    note: str


def ensure_dirs() -> None:
    ROBOCASA_DIR.mkdir(parents=True, exist_ok=True)


def status() -> RoboCasaStatus:
    ensure_dirs()
    xmls = iter_asset_files(ROBOCASA_DIR)
    if not xmls:
        return RoboCasaStatus(
            root=ROBOCASA_DIR,
            available=False,
            n_xml=0,
            note="未找到 XML。运行: python scripts/fetch_robocasa_assets.py --from-env/--link",
        )
    return RoboCasaStatus(
        root=ROBOCASA_DIR,
        available=True,
        n_xml=len(xmls),
        note=f"已发现 {len(xmls)} 个 XML/MJCF",
    )


def scan_and_register(registry: AssetRegistry) -> list[AssetSpec]:
    """扫描 assets/robocasa（含 softlink）下的模型并写入 registry。"""
    ensure_dirs()
    added: list[AssetSpec] = []
    for path in sorted(iter_asset_files(ROBOCASA_DIR)):
        try:
            rel = path.resolve().relative_to(ROBOCASA_DIR.resolve()).as_posix()
        except ValueError:
            # softlink 指向仓库外时，用相对 ROBOCASA_DIR 的展示路径
            rel = str(path)
        low = rel.lower()
        if "fixture" in low:
            category, prefix = "fixture", "robocasa.fixture"
        elif "/objects/" in low or low.startswith("objects"):
            category, prefix = "object", "robocasa.object"
        else:
            category, prefix = "robocasa", "robocasa.asset"
        asset_id = f"{prefix}.{path.stem}"
        if asset_id in registry.assets:
            asset_id = f"{asset_id}_{abs(hash(rel)) % 10_000:04d}"
        spec = AssetSpec(
            asset_id=asset_id,
            category=category,
            path=path,
            tags=("robocasa",),
            note=f"RoboCasa local: {rel}",
        )
        try:
            registry.register(spec)
            added.append(spec)
        except ValueError:
            continue
    _write_manifest(added)
    return added


def _write_manifest(added: list[AssetSpec]) -> None:
    ensure_dirs()
    payload = {
        "root": str(ROBOCASA_DIR),
        "count": len(added),
        "assets": [
            {
                "id": a.asset_id,
                "category": a.category,
                "path": str(a.path),
            }
            for a in added[:500]  # manifest 截断，避免巨型文件
        ],
        "assets_truncated": len(added) > 500,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def find_cabinet_candidate(registry: AssetRegistry | None = None) -> AssetSpec | None:
    """优先找名称含 cabinet/hinge 的 RoboCasa fixture；否则 None。"""
    from asset_registry import default_registry

    reg = registry or default_registry(scan_robocasa=False)
    scan_and_register(reg)
    preferred = []
    for spec in reg.by_category("fixture"):
        if not spec.path or not spec.path.exists():
            continue
        if "robocasa" not in spec.tags:
            continue
        name = spec.path.stem.lower()
        score = 0
        if "cabinet" in name:
            score += 2
        if "hinge" in name or "door" in name:
            score += 1
        if score:
            preferred.append((score, spec))
    if not preferred:
        return None
    preferred.sort(key=lambda x: (-x[0], x[1].asset_id))
    return preferred[0][1]


def _replace_with_symlink(dst: Path, src: Path) -> None:
    if not src.exists():
        return
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        if dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
        else:
            # 非空本地目录：挪到 .bak 再链（避免静默跳过）
            bak = dst.with_name(dst.name + ".local_bak")
            if bak.exists():
                shutil.rmtree(bak) if bak.is_dir() and not bak.is_symlink() else bak.unlink()
            dst.rename(bak)
    dst.symlink_to(src, target_is_directory=True)


def link_from_robocasa_package() -> Path | None:
    """若已安装 robocasa，把 models/assets 子目录链到 home_scene/assets/robocasa。"""
    src: Path | None = None
    try:
        import robocasa

        src = Path(robocasa.__path__[0]) / "models" / "assets"
    except Exception:
        src = None
    if src is None or not src.is_dir():
        if DEFAULT_WS_ASSETS.is_dir():
            src = DEFAULT_WS_ASSETS
        else:
            return None
    ensure_dirs()
    for name in ("fixtures", "objects", "textures", "generative_textures"):
        _replace_with_symlink(ROBOCASA_DIR / name, src / name)
    # 同时保留整包 external 指针，便于一眼看到来源
    _replace_with_symlink(ROBOCASA_DIR / "external", src)
    return ROBOCASA_DIR


def copy_tree_into(src: Path, dst_name: str) -> Path:
    """把外部目录复制/合并进 assets/robocasa/<dst_name>。"""
    ensure_dirs()
    dst = ROBOCASA_DIR / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def env_override_root() -> Path | None:
    raw = os.environ.get("HOME_SCENE_ROBOCASA_ROOT") or os.environ.get("ROBOCASA_ASSETS")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def link_from_root(src: Path) -> Path:
    """把给定 assets 根目录的子树软链进 home_scene/assets/robocasa。"""
    ensure_dirs()
    for name in ("fixtures", "objects", "textures", "generative_textures"):
        _replace_with_symlink(ROBOCASA_DIR / name, src / name)
    _replace_with_symlink(ROBOCASA_DIR / "external", src)
    return ROBOCASA_DIR
