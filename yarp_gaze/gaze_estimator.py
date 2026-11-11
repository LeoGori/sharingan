#
# Single-person gaze estimation with Sharingan, replicating the inference pipeline of demo.py
# (head detection with SAM3 or YOLOv5 -> most confident head -> Sharingan gaze point + inout).
# This file has no YARP dependency, so it can be used and tested on its own.
#

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from termcolor import colored

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.modeling.sharingan import Sharingan  # noqa: E402
from src.utils.common import spatial_argmax2d, square_bbox  # noqa: E402


# =============================== GLOBALS =============================== #
TERM_COLOR = "cyan"
DET_THR = 0.4  # head detection threshold (same as demo.py)
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

DEFAULT_CKPT_PATH = REPO_ROOT / "checkpoints" / "gf+vat+ours-head-bg-swap-mg0.ckpt"
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "conf" / "demo.yaml"


def _resolve_path(path):
	"""Resolve a relative path from the config against the repo root, so the module can run from any directory."""
	path = Path(path).expanduser()
	return str(path if path.is_absolute() else REPO_ROOT / path)


# ============================ HEAD DETECTORS ============================ #
def _empty_detections():
	return np.empty((0, 6), dtype=np.float32)


def _results_to_detections(result, class_id=0):
	if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
		return _empty_detections()
	boxes_xyxy = result.boxes.xyxy.cpu().numpy()
	boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes_xyxy))
	class_ids = np.full((len(boxes_xyxy), 1), class_id, dtype=np.float32)
	return np.concatenate([boxes_xyxy, boxes_conf[:, None], class_ids], axis=1).astype(np.float32)


class YoloHeadDetector:
	def __init__(self, config, device):
		ckpt_path = _resolve_path(config.get("weights", "./weights/yolov5m_crowdhuman.pt"))
		self.imgsz = int(config.get("imgsz", 640))
		self.conf = float(config.get("conf", 0.25))
		self.model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False)
		self.model.conf = self.conf
		self.model.iou = float(config.get("iou", 0.45))
		self.model.classes = config.get("classes", [1])
		self.model.amp = bool(config.get("amp", False))
		self.model = self.model.to(device)
		self.model.eval()

	def detect(self, image_rgb):
		"""image_rgb: (h, w, 3) uint8 RGB array. Returns (n, 6) array [x1, y1, x2, y2, conf, cls]."""
		results = self.model(image_rgb, size=self.imgsz).pred[0].cpu().numpy()
		results = results[results[:, 4] >= self.conf] if results.size > 0 else results
		if results.size == 0:
			return _empty_detections()
		detections = np.zeros((len(results), 6), dtype=np.float32)
		detections[:, :5] = results[:, :5]
		return detections


class SAMHeadDetector:
	def __init__(self, config, device):
		from ultralytics.models.sam import SAM3SemanticPredictor

		overrides = dict(
			conf=float(config.get("conf", 0.20)),
			task=str(config.get("task", "segment")),
			mode=str(config.get("mode", "predict")),
			imgsz=int(config.get("imgsz", 644)),
			model=_resolve_path(config.get("model", "./weights/sam3.1_multiplex.pt")),
			half=bool(config.get("half", False)),
			save=bool(config.get("save", False)),
			verbose=bool(config.get("verbose", False)),
		)
		if config.get("iou", None) is not None:
			overrides["iou"] = float(config.get("iou"))
		overrides["device"] = str(config.get("device", device))
		print(colored(f"Using SAM3 with the following overrides: {overrides}", TERM_COLOR))

		self.prompts = list(config.get("prompts", ["face"]))
		self.predictor = SAM3SemanticPredictor(overrides=overrides)

	def detect(self, image_rgb):
		"""image_rgb: (h, w, 3) uint8 RGB array. Returns (n, 6) array [x1, y1, x2, y2, conf, cls]."""
		# SAM (ultralytics) expects numpy images in BGR order, as read by cv2
		self.predictor.set_image(np.ascontiguousarray(image_rgb[..., ::-1]))
		results = self.predictor(text=self.prompts)
		result = results[0] if isinstance(results, list) else results
		return _results_to_detections(result)


def load_head_detector(config, device):
	config = config or {}
	detector_type = str(config.get("type", "yolo")).lower()
	if detector_type == "yolo":
		return YoloHeadDetector(config.get("yolo", {}), device)
	if detector_type == "sam":
		return SAMHeadDetector(config.get("sam", {}), device)
	raise ValueError(f"Unsupported head detector type: {detector_type}")


# ============================== SHARINGAN =============================== #
def load_sharingan_model(ckpt_path, device):
	sharingan = Sharingan(
		patch_size=16,
		token_dim=768,
		image_size=224,
		gaze_feature_dim=512,
		encoder_depth=12,
		encoder_num_heads=12,
		encoder_num_global_tokens=0,
		encoder_mlp_ratio=4.0,
		encoder_use_qkv_bias=True,
		encoder_drop_rate=0.0,
		encoder_attn_drop_rate=0.0,
		encoder_drop_path_rate=0.0,
		decoder_feature_dim=128,
		decoder_hooks=[2, 5, 8, 11],
		decoder_hidden_dims=[48, 96, 192, 384],
		decoder_use_bn=True,
	)
	checkpoint = torch.load(ckpt_path, map_location="cpu")
	checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
	sharingan.load_state_dict(checkpoint, strict=True)
	print(colored(f"Successfully loaded Sharingan weights from {ckpt_path}.", TERM_COLOR))
	sharingan.eval()
	sharingan.to(device)
	return sharingan


class SharinganGazeEstimator:
	"""Estimates the gaze point and inout score of the most confident detected head in an RGB image."""

	def __init__(self, ckpt_path=DEFAULT_CKPT_PATH, config_path=DEFAULT_CONFIG_PATH, det_thr=DET_THR, device=None):
		self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.det_thr = det_thr
		config = OmegaConf.load(str(config_path))
		self.head_detector = load_head_detector(config.get("head_detector", {}), self.device)
		self.sharingan = load_sharingan_model(str(ckpt_path), self.device)

	@torch.no_grad()
	def predict(self, image_rgb):
		"""
		Args:
			image_rgb: (h, w, 3) uint8 RGB array.
		Returns:
			(gaze_x, gaze_y, inout, head_bbox, head_conf) with the gaze point normalized to [0, 1] w.r.t. the image
			size, the inout probability in [0, 1], the detected head box (x1, y1, x2, y2) in pixels and its detection
			score, or None if no head is detected above the detection threshold.
		"""
		img_h, img_w = image_rgb.shape[:2]

		# 1. Detect heads and keep the most confident one (single-person pipeline, as in demo.py)
		detections = self.head_detector.detect(image_rgb)
		detections = detections[detections[:, 4] > self.det_thr]
		if len(detections) == 0:
			return None
		best = detections[np.argmax(detections[:, 4])]
		head_bboxes = torch.from_numpy(best[:4][None, :].copy()).float()  # (1, 4)
		t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)

		# 2. Extract and transform heads
		image = Image.fromarray(image_rgb)
		heads = torch.stack([TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224)) for bbox in t_head_bboxes])
		heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

		# 3. Transform image
		image = TF.normalize(TF.resize(TF.to_tensor(image), (224, 224)), mean=IMG_MEAN, std=IMG_STD)

		# 4. Normalize head bboxes
		t_head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

		# 5. Predict gaze
		sample = {
			"image": image.unsqueeze(0).to(self.device),  # (1, 3, 224, 224)
			"heads": heads.unsqueeze(0).to(self.device),  # (1, 1, 3, 224, 224)
			"head_bboxes": t_head_bboxes.unsqueeze(0).to(self.device),  # (1, 1, 4)
		}
		_, gaze_heatmaps, inouts = self.sharingan(sample)
		gaze_point = spatial_argmax2d(gaze_heatmaps.squeeze(0).cpu(), normalize=True)[0]  # (2,)
		inout = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()[0]
		head_bbox = tuple(float(v) for v in best[:4])
		return float(gaze_point[0]), float(gaze_point[1]), float(inout), head_bbox, float(best[4])
