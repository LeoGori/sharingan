import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def to_list(x):
    """Converte tensori / numpy array / liste in liste Python serializzabili JSON."""
    if x is None:
        return None

    if hasattr(x, "cpu"):
        x = x.cpu()

    if hasattr(x, "numpy"):
        x = x.numpy()

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, list):
        return x

    return x


def serialize_result(result, image_path):
    """
    Estrae boxes, classi, confidenze e poligoni delle maschere
    da un oggetto Results di Ultralytics.
    """
    item = {
        "image": str(image_path),
        "width": None,
        "height": None,
        "detections": [],
    }

    # Dimensioni immagine
    if hasattr(result, "orig_shape") and result.orig_shape is not None:
        h, w = result.orig_shape[:2]
        item["height"] = int(h)
        item["width"] = int(w)

    names = getattr(result, "names", {}) or {}

    boxes_xyxy = []
    boxes_conf = []
    boxes_cls = []

    if result.boxes is not None:
        if result.boxes.xyxy is not None:
            boxes_xyxy = to_list(result.boxes.xyxy)

        if result.boxes.conf is not None:
            boxes_conf = to_list(result.boxes.conf)

        if result.boxes.cls is not None:
            boxes_cls = to_list(result.boxes.cls)

    masks_xy = []
    masks_xyn = []

    if result.masks is not None:
        # Poligoni in pixel
        if hasattr(result.masks, "xy"):
            masks_xy = [
                np.asarray(poly).tolist()
                for poly in result.masks.xy
            ]

        # Poligoni normalizzati 0-1
        if hasattr(result.masks, "xyn"):
            masks_xyn = [
                np.asarray(poly).tolist()
                for poly in result.masks.xyn
            ]

    num_instances = max(
        len(boxes_xyxy),
        len(masks_xy),
        len(masks_xyn),
        len(boxes_cls),
        len(boxes_conf),
    )

    for i in range(num_instances):
        cls_id = None
        cls_name = None

        if i < len(boxes_cls):
            cls_id = int(boxes_cls[i])

            if isinstance(names, dict):
                cls_name = names.get(cls_id, str(cls_id))
            elif isinstance(names, list):
                cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            else:
                cls_name = str(cls_id)

        detection = {
            "index": i,
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": float(boxes_conf[i]) if i < len(boxes_conf) else None,
            "bbox_xyxy": boxes_xyxy[i] if i < len(boxes_xyxy) else None,
            # "mask_polygon_xy": masks_xy[i] if i < len(masks_xy) else None,
            # "mask_polygon_xyn": masks_xyn[i] if i < len(masks_xyn) else None,
        }

        item["detections"].append(detection)

    return item


def find_images(input_dir):
    input_dir = Path(input_dir)

    images = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Cartella contenente i frame / immagini.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Cartella dove salvare frame annotati e JSON.",
    )

    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help='Prompt testuali, esempio: --text person bus glasses',
    )

    parser.add_argument(
        "--model",
        default="/home/icub/code/sam3.1/sam3.1_multiplex.pt",
        help="Path o nome del modello SAM3.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Soglia di confidenza.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=644,
        help="Dimensione immagine per inferenza.",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="Usa FP16.",
    )

    parser.add_argument(
        "--device",
        default="cuda:1",
        help='Device da usare, esempio: "0", "cuda:0", "cpu".',
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Se impostato, salva anche i file JSON con i risultati delle segmentazioni.",
    )

    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Se impostato, salva le immagini annotate.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    annotated_root = output_dir / "annotated"
    if args.save_annotated:
        annotated_root.mkdir(parents=True, exist_ok=True)

    json_root = None
    if args.save_json:
        json_root = output_dir / "json"
        json_root.mkdir(parents=True, exist_ok=True)
    images = find_images(input_dir)

    if len(images) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in: {input_dir}")

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        imgsz=args.imgsz,
        model=args.model,
        half=args.half,
        save=False,
        verbose=False,
    )

    if args.device is not None:
        overrides["device"] = args.device

    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Raggruppo i risultati per cartella relativa
    results_by_folder = {}

    for image_path in tqdm(images, desc="Processing frames"):
        rel_path = image_path.relative_to(input_dir)
        rel_folder = rel_path.parent

        out_folder = annotated_root / rel_folder

        out_image_path = out_folder / image_path.name

        # SAM3 su singola immagine
        predictor.set_image(str(image_path))
        results = predictor(text=args.text)

        # Normalmente results è una lista con un solo elemento
        result = results[0] if isinstance(results, list) else results

        # Salva immagine annotata
        annotated = result.plot()
        if args.save_annotated:
            out_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_image_path), annotated)

        # Serializza risultati
        if args.save_json:
            serialized = serialize_result(result, image_path)

            serialized["annotated_image"] = str(out_image_path)
            serialized["text_prompts"] = args.text

            folder_key = str(rel_folder)

            if folder_key not in results_by_folder:
                results_by_folder[folder_key] = []

            results_by_folder[folder_key].append(serialized)

    # Scrive un JSON per ogni cartella
    if args.save_json:
        for folder_key, folder_results in results_by_folder.items():
            json_folder = json_root / folder_key
            json_folder.mkdir(parents=True, exist_ok=True)

            json_path = json_folder / "segmentations.json"

            payload = {
                "input_folder": str(input_dir / folder_key),
                "num_images": len(folder_results),
                "text_prompts": args.text,
                "model": args.model,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "results": folder_results,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"JSON salvato: {json_path}")

    print(f"Frame annotati salvati in: {output_dir}")


if __name__ == "__main__":
    main()