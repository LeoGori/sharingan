#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

"""Visualize per-image annotations from a CSV-style file.

The script reads annotations with columns like:

image_path,id,gaze_x,gaze_y,eye_bbox_x_min,eye_bbox_y_min,eye_bbox_x_max,eye_bbox_y_max,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,in_or_out

It filters the rows that match a requested image path and overlays the body box,
head box, eye location, and gaze point on the image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


DEFAULT_COLUMNS = [
    "image_path",
    "id",
    "gaze_x",
    "gaze_y",
    "eye_bbox_x_min",
    "eye_bbox_y_min",
    "eye_bbox_x_max",
    "eye_bbox_y_max",
    "head_bbox_x_min",
    "head_bbox_y_min",
    "head_bbox_x_max",
    "head_bbox_y_max",
    "in_or_out"
]

LEGACY_COLUMNS = [column for column in DEFAULT_COLUMNS if column != "origin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize annotations for a single image.")
    parser.add_argument("--annotations", required=True, type=Path, help="Path to the CSV or TXT annotation file.")
    parser.add_argument("--image", required=True, type=Path, help="Image name or path to visualize.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root searched recursively when --image is only a file name.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path. If omitted, saves next to the image with a suffix.")
    parser.add_argument("--show", action="store_true", help="Display the annotated image in a window.")
    parser.add_argument("--exact-match-only", action="store_true", help="Only match rows whose image name exactly equals the provided image name.")
    return parser.parse_args()


def load_annotations(path: Path) -> pd.DataFrame:
    for columns in (DEFAULT_COLUMNS, LEGACY_COLUMNS):
        try:
            frame = pd.read_csv(path, header=None, names=columns, index_col=False)
        except pd.errors.ParserError:
            continue

        if not frame.empty and str(frame.iloc[0]["image_path"]).endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
            return frame

    raise ValueError(f"Could not parse annotations in {path} using the expected CSV layouts.")


def resolve_image_path(raw_path: str, annotation_file: Path, image_root: Optional[Path]) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if image_root is not None:
        rooted = image_root / candidate
        if rooted.exists():
            return rooted

    relative_to_annotations = (annotation_file.parent / candidate).resolve()
    if relative_to_annotations.exists():
        return relative_to_annotations

    return candidate


def resolve_input_image(image: Path, image_root: Optional[Path]) -> Path:

    print(f"Resolving image path for {image} with image_root={image_root}")
    if image.exists():
        return image.resolve()

    search_root = image_root if image_root is not None else Path.cwd()
    matches = [path for path in search_root.rglob(image.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find image named {image.name}. Provide --image-root if the file is not in the current directory."
    )


def pick_rows(frame: pd.DataFrame, image: Path, annotation_file: Path, image_root: Optional[Path], exact_match_only: bool) -> pd.DataFrame:
    # image_name = image.name
    image_path = str(image)
    # image_stem = image.stem

    def matches(value: object) -> bool:
        text = str(value)
        # print(1, text)
        # print(2, image_path)
        # candidate_name = Path(text).name
        candidate_name = text
        # print(3, candidate_name)
        # print(4, candidate_name == image_path)
        if exact_match_only:
            # return candidate_name == image_name or candidate_name == f"{image_stem}{Path(image_name).suffix}"
            return candidate_name == image_path
        else:
            return candidate_name == image_path

        # resolved = resolve_image_path(text, annotation_file, image_root)
        # return candidate_name == image_name or resolved.name == image_name

    mask = frame["image_path"].apply(matches)
    return frame[mask].copy()


def draw_box(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], label: str) -> None:
    height, width = canvas.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
        top = max(0, y1 - text_h - 8)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, top + text_h + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 3, top + text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_point(canvas: np.ndarray, x: int, y: int, color: tuple[int, int, int], label: str, radius: int = 5) -> None:
    height, width = canvas.shape[:2]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    cv2.circle(canvas, (x, y), radius, color, -1)
    if label:
        cv2.putText(canvas, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def to_int(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def to_pixel(value: object, scale: int) -> Optional[int]:
    if pd.isna(value):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= numeric <= 1.0:
        numeric *= scale

    return int(round(numeric))


def annotate_image(image: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    canvas = image.copy()
    img_h, img_w = canvas.shape[:2]

    for index, row in rows.reset_index(drop=True).iterrows():
        sample_id = row.get("id", index)
        in_or_out = row.get("in_or_out", "")
        meta = row.get("meta", "")
        prefix = f"id={sample_id}"
        if not pd.isna(in_or_out):
            prefix += f" in_or_out={in_or_out}"
        if not pd.isna(meta) and str(meta):
            prefix += f" | {meta}"

        body_x1 = to_pixel(row.get("eye_bbox_x_min"), img_w)
        body_y1 = to_pixel(row.get("eye_bbox_y_min"), img_h)
        body_x2 = to_pixel(row.get("eye_bbox_x_max"), img_w)
        body_y2 = to_pixel(row.get("eye_bbox_y_max"), img_h)
        if None not in (body_x1, body_y1, body_x2, body_y2):
            draw_box(canvas, body_x1, body_y1, body_x2, body_y2, (255, 180, 0), f"eyes {prefix}")

        head_x1 = to_int(row.get("head_bbox_x_min"))
        head_y1 = to_int(row.get("head_bbox_y_min"))
        head_x2 = to_int(row.get("head_bbox_x_max"))
        head_y2 = to_int(row.get("head_bbox_y_max"))
        if None not in (head_x1, head_y1, head_x2, head_y2):
            draw_box(canvas, head_x1, head_y1, head_x2, head_y2, (0, 0, 255), f"head {sample_id}")

        gaze_x = to_pixel(row.get("gaze_x"), img_w)
        gaze_y = to_pixel(row.get("gaze_y"), img_h)


        if None not in (gaze_x, gaze_y):
            draw_point(canvas, gaze_x, gaze_y, (0, 255, 255), "gaze", radius=6)

    return canvas


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotations)

    image_path = resolve_input_image(args.image, args.image_root)

    print(f"Resolved image path: {image_path}")
    print(f"Using annotation file: {args.annotations}")

    rows = pick_rows(annotations, image_path, args.annotations, args.image_root, args.exact_match_only)
    if rows.empty:
        available = annotations["image_path"].astype(str).head(5).tolist()
        raise ValueError(
            f"No annotations found for {image_path}. "
            f"Matching is done by image name. Example annotation paths: {available}"
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    canvas = annotate_image(image, rows)

    output_path = args.output
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved annotated image to {output_path}")
    print(f"Matched {len(rows)} annotation row(s)")

    if args.show:
        try:
            cv2.imshow("annotations", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Skipping interactive display because OpenCV was built without GUI support.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())