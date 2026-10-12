"""Style presets: one folder per style under styles/, holding everything specific to that style.

    styles/<name>/style.toml    prompt, look and option defaults (see styles/crayon-pastel/style.toml)
    styles/<name>/palettes.txt  optional per-frame colour/lighting lines
    styles/<name>/refs/         style reference images (gitignored)
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

STYLES_DIR = Path(__file__).resolve().parent.parent / "styles"
STRUCTURES = ["photo", "lineart", "depth"]
# Defaults that name files, resolved relative to the preset folder.
PATH_DEFAULTS = {"palettes", "style_dir"}


@dataclass
class Style:
    name: str
    dir: Path
    description: str = ""
    prompt: str = ""
    look: dict[str, str] = field(default_factory=dict)  # per structure
    defaults: dict = field(default_factory=dict)

    @property
    def refs_dir(self) -> Path | None:
        refs = self.dir / "refs"
        return refs if refs.is_dir() else None


def load(name_or_path: str | Path) -> Style:
    """Load a preset by name (a folder under styles/) or by path to its folder."""
    d = Path(name_or_path)
    if not (d / "style.toml").exists():
        d = STYLES_DIR / str(name_or_path)
    if not (d / "style.toml").exists():
        names = sorted(p.parent.name for p in STYLES_DIR.glob("*/style.toml"))
        raise SystemExit(f"No style preset {name_or_path!r}; available: {', '.join(names)}")
    cfg = tomllib.loads((d / "style.toml").read_text())
    look = cfg.get("look", "")
    if isinstance(look, str):
        look = dict.fromkeys(STRUCTURES, look)
    defaults = {k: d / v if k in PATH_DEFAULTS else v for k, v in cfg.get("defaults", {}).items()}
    return Style(d.name, d, cfg.get("description", ""), cfg.get("prompt", "").strip(), look, defaults)
