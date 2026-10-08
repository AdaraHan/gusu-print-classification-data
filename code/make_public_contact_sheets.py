"""Make caption-safe thumbnails from published images without cropping artwork."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make(rows, title, path):
    cols, width, height = 4, 240, 210
    page = Image.new("RGB", (cols * width, 42 + ((len(rows) + cols - 1) // cols) * height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((8, 8), title, fill="black")
    for position, row in enumerate(rows):
        x, y = position % cols * width, 42 + position // cols * height
        with Image.open(ROOT / row["image_path"]) as source:
            image = ImageOps.contain(source.convert("RGB"), (width - 16, height - 62), Image.Resampling.LANCZOS)
        page.paste(image, (x + (width - image.width) // 2, y + 4 + (height - 62 - image.height) // 2))
        draw.text((x + 8, y + height - 51), f"{row['sample_id']}  fold {row['fold']}", fill="black")
        draw.text((x + 8, y + height - 32), f"true {row['true_class']} / pred {row['pred_class']}", fill="black")
    page.save(path, format="PNG")


def main():
    rows = read(ROOT / "predictions/effnetv2_v4_misclassified.csv")
    negatives = sorted((row for row in rows if row["true_class"] == "gusu"), key=lambda row: row["sample_id"])
    positives = sorted((row for row in rows if row["true_class"] == "non_gusu"), key=lambda row: row["sample_id"])
    if len(negatives) != 18 or len(positives) != 28:
        raise ValueError("unexpected EfficientNet error counts")
    OUT.mkdir(exist_ok=True)
    for name in ("effnet_false_negatives_public.png", "effnet_false_positives_public.png"):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    make(negatives, "EfficientNetV2-S OOF false negatives (18)", OUT / "effnet_false_negatives_public.png")
    make(positives, "EfficientNetV2-S OOF false positives (28)", OUT / "effnet_false_positives_public.png")
    print("contact_sheets=2 panels=18+28 source_images_unchanged=true")


if __name__ == "__main__":
    main()
