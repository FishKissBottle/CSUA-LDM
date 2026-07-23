from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "论文章节" / "与其他模型对比的定性展示"
SOURCE_ROOT = Path(os.environ.get("CSUA_LDM_SOTA_SOURCE_ROOT", str(FIGURE_ROOT)))
DATASETS = ("oli2msi", "sen2naip", "fy3fs2")
ROW_GAP = 15
TRAILING_WHITE_SPACE = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trim_bottom_whitespace(image: Image.Image) -> Image.Image:
    array = np.asarray(image)
    white_fraction = (array.min(axis=2) >= 248).mean(axis=1)
    content_rows = np.flatnonzero(white_fraction < 0.985)
    if content_rows.size == 0:
        return image.copy()
    bottom = min(int(content_rows[-1]) + 1 + TRAILING_WHITE_SPACE, image.height)
    return image.crop((0, 0, image.width, bottom))


def _combine(figure_type: str) -> Path:
    source_directory = SOURCE_ROOT / figure_type
    output_directory = FIGURE_ROOT / figure_type
    input_paths = [
        source_directory / f"{dataset}_{figure_type}_Qualitative_Comparison.jpg"
        for dataset in DATASETS
    ]
    hashes_before = {path: _sha256(path) for path in input_paths}

    first_sample_rows: list[Image.Image] = []
    for path in input_paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            first_sample = rgb.crop((0, 0, rgb.width, rgb.height // 2))
            first_sample_rows.append(_trim_bottom_whitespace(first_sample))

    canvas_width = max(row.width for row in first_sample_rows)
    canvas_height = sum(row.height for row in first_sample_rows)
    canvas_height += ROW_GAP * (len(first_sample_rows) - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    y = 0
    for index, row in enumerate(first_sample_rows):
        x = (canvas_width - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height
        if index < len(first_sample_rows) - 1:
            y += ROW_GAP

    output_canvas = canvas
    if figure_type == "Spatial":
        canvas_array = np.asarray(canvas)
        content_columns = np.flatnonzero(
            (canvas_array.min(axis=2) < 245).any(axis=0)
        )
        if content_columns.size:
            right = min(int(content_columns[-1]) + 13, canvas.width)
            output_canvas = canvas.crop((0, 0, right, canvas.height))

    output_path = output_directory / f"SOTA_{figure_type}_Qualitative_Comparison_First_Samples.jpg"
    output_canvas.save(output_path, quality=95, subsampling=0, dpi=(300, 300))
    if output_canvas is not canvas:
        output_canvas.close()
    canvas.close()
    for row in first_sample_rows:
        row.close()

    hashes_after = {path: _sha256(path) for path in input_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("An original qualitative comparison figure was modified unexpectedly.")
    return output_path


def main() -> None:
    for figure_type in ("Spatial", "Spectral"):
        output_path = _combine(figure_type)
        print(output_path)


if __name__ == "__main__":
    main()
