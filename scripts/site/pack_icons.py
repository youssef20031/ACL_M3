"""Second half of scripts/site/render.mjs: turns the rendered PNGs into the files in public/.

    python scripts/site/pack_icons.py <render-dir>

<render-dir>/manifest.json (written by render.mjs) lists which images are opaque, which to
copy, and which frames go into the ICO. Opaque images are saved as RGB (no alpha channel:
apple-touch-icon must not have one, and it makes og-image.png smaller); the ICO keeps the
browser-rendered 16/32/48 frames instead of downscaling one large image.
"""

import json
import sys
from pathlib import Path

from PIL import Image


def main(work: Path) -> None:
    manifest = json.loads((work / "manifest.json").read_text())
    out = Path(manifest["out"])
    opaque = set(manifest["opaque"])

    for name in manifest["copy"]:
        img = Image.open(work / name)
        img.load()
        if name in opaque:
            img = img.convert("RGB")
        img.save(out / name, optimize=True)
        print(f"wrote {out / name} {img.size[0]}x{img.size[1]} {img.mode}")

    frames = [Image.open(work / name).convert("RGBA") for name in manifest["ico"]["frames"]]
    frames.sort(key=lambda f: f.size[0])
    ico_path = out / manifest["ico"]["file"]
    largest = frames[-1]
    largest.save(
        ico_path,
        format="ICO",
        sizes=[f.size for f in frames],
        append_images=frames[:-1],
    )
    with Image.open(ico_path) as check:
        sizes = sorted(check.info.get("sizes", set()))
    print(f"wrote {ico_path} sizes={sizes}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
