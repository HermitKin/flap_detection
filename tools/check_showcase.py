"""Verify public documentation links, images, GIF frames and demo result records."""

from pathlib import Path
import json
import re
from urllib.parse import unquote

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "docs/USAGE.md", ROOT / "docs/ALGORITHM.md",
             ROOT / "docs/assets/README.md"]
REQUIRED_IMAGES = ["ui-detection.png", "ui-keypoints.png", "ui-reading.png", "ui-hsv.png",
                   "pipeline.png", "pipeline.gif", "level-sweep.gif", "architecture.png", "algorithm.png"]


def check():
    errors = []
    for document in DOCUMENTS:
        content = document.read_text(encoding="utf-8")
        if content.count("```") % 2:
            errors.append(f"Unbalanced code fence: {document.name}")
        # Paths inside fenced examples are not hyperlinks.
        prose = re.sub(r"```.*?```", "", content, flags=re.S)
        for destination in re.findall(r"\]\(([^)]+)\)", prose):
            if destination.startswith(("https://", "http://", "#")):
                continue
            relative = unquote(destination.split("#", 1)[0])
            if not (document.parent / relative).exists():
                errors.append(f"Broken local link in {document.name}: {relative}")
    for name in REQUIRED_IMAGES:
        path = ROOT / "docs/assets" / name
        if not path.exists():
            errors.append(f"Missing image: {name}")
            continue
        with Image.open(path) as image:
            if image.width < 600 or image.height < 350:
                errors.append(f"Unexpected small image: {name}")
            count = getattr(image, "n_frames", 1)
            if path.suffix == ".gif" and (count < 4 or image.info.get("loop") != 0):
                errors.append(f"GIF must contain at least four looping frames: {name}")
            for frame in range(count):
                image.seek(frame)
                image.load()
            print(f"Image OK: {name}, {image.size}, frames={count}")
    results = json.loads((ROOT / "docs/assets/synthetic-results.json").read_text(encoding="utf-8"))
    if len(results["cases"]) != 6 or any(c["absolute_error"] >= 1.5 for c in results["cases"]):
        errors.append("Unexpected synthetic smoke-test results.")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Showcase links and assets OK")


if __name__ == "__main__":
    check()
