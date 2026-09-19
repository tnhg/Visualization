"""Create an exact visual index of predicted-versus-true Grad-CAM overlays."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _resize(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert('RGB')
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    panel = Image.new('RGB', (width, height), 'white')
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return panel


def build_contact_sheet(gradcam_dir: Path, output: Path, columns: int) -> int:
    pairs = []
    for pred in sorted(gradcam_dir.glob('*_pred_overlay.png')):
        stem = pred.name.removesuffix('_pred_overlay.png')
        true = gradcam_dir / f'{stem}_true_overlay.png'
        if true.is_file():
            pairs.append((stem, pred, true))
    if not pairs:
        raise RuntimeError(f'No matched *_pred_overlay.png / *_true_overlay.png pairs in {gradcam_dir}')

    image_width, image_height = 224, 224
    padding, title_height, label_height = 16, 34, 20
    tile_width = image_width * 2 + padding * 3
    tile_height = title_height + label_height + image_height + padding * 2
    rows = (len(pairs) + columns - 1) // columns
    header_height = 48
    canvas = Image.new(
        'RGB',
        (columns * tile_width + (columns + 1) * padding, header_height + rows * (tile_height + padding) + padding),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, header_font = _font(15), _font(13), _font(20)
    draw.text(
        (padding, padding),
        'Grad-CAM overlays: predicted class (left) vs ground-truth class (right)',
        fill='black',
        font=header_font,
    )
    for index, (stem, pred_path, true_path) in enumerate(pairs):
        row, column = divmod(index, columns)
        left = padding + column * (tile_width + padding)
        top = header_height + padding + row * (tile_height + padding)
        draw.rectangle((left, top, left + tile_width, top + tile_height), outline='#b0b0b0', width=1)
        draw.text((left + padding, top + 5), stem.replace('_', ' '), fill='black', font=title_font)
        image_top = top + title_height + label_height
        draw.text((left + padding, top + title_height), 'Predicted CAM', fill='#8b0000', font=label_font)
        draw.text((left + padding * 2 + image_width, top + title_height), 'True CAM', fill='#004c99', font=label_font)
        canvas.paste(_resize(Image.open(pred_path), image_width, image_height), (left + padding, image_top))
        canvas.paste(_resize(Image.open(true_path), image_width, image_height), (left + padding * 2 + image_width, image_top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gradcam-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--columns', type=int, default=2)
    args = parser.parse_args()
    if args.columns < 1:
        raise ValueError('--columns must be positive')
    count = build_contact_sheet(args.gradcam_dir, args.output, args.columns)
    print(f'Wrote {args.output} with {count} predicted/true Grad-CAM pairs.')


if __name__ == '__main__':
    main()
