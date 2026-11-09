#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
import sys
import shlex
import shutil
import argparse
import importlib
import datetime as dt
from pathlib import Path
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import subprocess as sp
from omegaconf import OmegaConf
from termcolor import colored

import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.modeling.sharingan import Sharingan
from src.utils.common import spatial_argmax2d, square_bbox

from boxmot import DeepOCSORT, BYTETracker, OCSORT


DEFAULT_DEMO_CONFIG_PATH = Path(__file__).resolve().parent / "src" / "conf" / "demo.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}



# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Predict gaze on videos")
parser.add_argument("--input-dir", type=str, default="data", help="Name of the folder where to find the input.")
parser.add_argument("--input-filename", type=str, help="Name of the clip file to process (with extension).")
parser.add_argument("--output-dir", type=str, default="data", help="Name of the folder where to save the output.")
parser.add_argument("--heatmap-pid", type=int, default=-1, help="pid of the person to draw the heatmap of.")
parser.add_argument("--config", type=str, default=str(DEFAULT_DEMO_CONFIG_PATH), help="Path to the demo configuration file.")

parser.add_argument("--filter-by-inout", action='store_true', help="Whether to hide the gaze point when inout < 0.5.")
parser.add_argument('--no-filter-by-inout', dest='filter_by_inout', action='store_false')
parser.set_defaults(filter_by_inout=False)

parser.add_argument("--show-gaze-vec", action='store_true', help="Whether to draw the gaze vector.")
parser.add_argument('--no-show-gaze-vec', dest='show_gaze_vec', action='store_false')
parser.set_defaults(show_gaze_vec=False)

args = parser.parse_args()


# =============================== GLOBALS =============================== #
TERM_COLOR = "cyan"
COLOR_NAMES = ["mediumvioletred", "green", "dodgerblue", "crimson", "goldenrod", "DarkSlateGray", 
			   "saddlebrown", "purple", "teal"]
COLORS = [(199, 21, 133), (0, 128, 0), (30, 144, 255), (220, 20, 60), (218, 165, 32), 
		  (47, 79, 79), (139, 69, 19), (128, 0, 128), (0, 128, 128)]

DET_THR = 0.4 # head detection threshold
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# CKPT_PATH = "checkpoints/videoattentiontarget.pt"
# CKPT_PATH = "checkpoints/best-v12.ckpt"
CKPT_PATH = "checkpoints/gf+vat+ours-head-bg-swap-mg0.ckpt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(colored(f"Using device: {DEVICE}", TERM_COLOR))

# ========================= UTILITY FUNCTIONS =========================== #
def expand_bbox(bbox, img_w, img_h, k=0.1):
	w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
	bbox[0] = max(0, bbox[0] - k * w)
	bbox[1] = max(0, bbox[1] - k * h)
	bbox[2] = min(img_w, bbox[2] + k * w)
	bbox[3] = min(img_h, bbox[3] + k * h)
	return bbox


def to_list(x):
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


def find_images(input_dir):
	input_dir = Path(input_dir)
	images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
	return sorted(images)

def load_tracker():
	#tracker = DeepOCSORT(
	#  model_weights=Path('/idiap/temp/stafasca/weights/tracking/osnet_x0_25_msmt17.pt'),  # which ReID model to use
	#  device=self.device,  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
	#  fp16=True,  # wether to run the ReID model with half precision or not
	#)
	#tracker = BYTETracker()
	tracker = OCSORT()
	return tracker

def _empty_detections():
	return np.empty((0, 6), dtype=np.float32)


def _results_to_detections(result, class_id=0):
	detections = []

	boxes_xyxy = to_list(result.boxes.xyxy) if result.boxes is not None and result.boxes.xyxy is not None else []
	boxes_conf = to_list(result.boxes.conf) if result.boxes is not None and result.boxes.conf is not None else []

	for index, bbox in enumerate(boxes_xyxy):
		confidence = float(boxes_conf[index]) if index < len(boxes_conf) else 1.0
		detections.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_id], dtype=np.float32))

	if not detections:
		return _empty_detections()

	return np.stack(detections)


class BaseHeadDetector:
	def detect(self, image, frame_path=None):
		raise NotImplementedError


class YoloHeadDetector(BaseHeadDetector):
	def __init__(self, config, device):
		ckpt_path = config.get("weights", "./weights/yolov5m_crowdhuman.pt")
		self.imgsz = int(config.get("imgsz", 640))
		self.conf = float(config.get("conf", 0.25))
		self.iou = float(config.get("iou", 0.45))
		self.classes = config.get("classes", [1])
		self.amp = bool(config.get("amp", False))
		self.model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False)
		self.model.conf = self.conf
		self.model.iou = self.iou
		self.model.classes = self.classes
		self.model.amp = self.amp
		self.model = self.model.to(device)
		self.model.eval()

	def detect(self, image, frame_path=None):
		results = self.model(image, size=self.imgsz).pred[0].cpu().numpy()
		if results.size == 0:
			return _empty_detections()

		results = results[results[:, 4] >= self.conf]
		if results.size == 0:
			return _empty_detections()

		detections = []
		for detection in results:
			bbox = detection[:4]
			confidence = detection[4]
			detections.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence, 0], dtype=np.float32))

		return np.stack(detections) if detections else _empty_detections()


class SAMHeadDetector(BaseHeadDetector):
	def __init__(self, config, device):
		try:
			from ultralytics.models.sam import SAM3SemanticPredictor
		except ImportError as exc:
			raise ImportError("SAM3 is not available. Install ultralytics with SAM support to use the SAM head detector.") from exc

		overrides = dict(
			conf=float(config.get("conf", 0.20)),
			task=str(config.get("task", "segment")),
			mode=str(config.get("mode", "predict")),
			imgsz=int(config.get("imgsz", 644)),
			model=str(config.get("model", "./weights/sam3.1_multiplex.pt")),
			half=bool(config.get("half", False)),
			save=bool(config.get("save", False)),
			verbose=bool(config.get("verbose", False)),
		)
		if config.get("iou", None) is not None:
			overrides["iou"] = float(config.get("iou"))
		overrides["device"] = str(config.get("device", device))

		# print overrides
		print(colored(f"Using SAM3 with the following overrides: {overrides}", TERM_COLOR))
		for key, value in overrides.items():
			print(colored(f"  {key}: {value}", TERM_COLOR))

		self.prompts = list(config.get("prompts", ["face"]))
		self.predictor = SAM3SemanticPredictor(overrides=overrides)

	def _set_image(self, image, frame_path=None):
		if frame_path is not None:
			self.predictor.set_image(str(frame_path))
			return

		image_np = np.asarray(image)
		try:
			self.predictor.set_image(image_np)
			return
		except Exception:
			pass

		with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
			image_bgr = image_np[..., ::-1] if image_np.ndim == 3 and image_np.shape[2] == 3 else image_np
			cv2.imwrite(tmp_file.name, image_bgr)
			temp_path = tmp_file.name

		try:
			self.predictor.set_image(temp_path)
		finally:
			os.unlink(temp_path)

	def detect(self, image, frame_path=None):
		self._set_image(image, frame_path=frame_path)
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

def load_sharingan_model(ckpt_path, device):
	# Build model
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

	# Load checkpoint
	checkpoint = torch.load(ckpt_path, map_location="cpu")
	checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
	sharingan.load_state_dict(checkpoint, strict=True)
	print(colored(f"Successfully loaded Sharingan weights from {ckpt_path}.", TERM_COLOR))
	sharingan.eval()
	sharingan.to(device)
	return sharingan

def predict_gaze(image, sharingan, head_detector, tracker=None, frame_path=None):
	# 1. Convert image
	image_np = np.array(image)
	img_h, img_w, img_c = image_np.shape
 
	raw_detections = head_detector.detect(image_np, frame_path=frame_path)
	detections = []
	for k, raw_detection in enumerate(raw_detections):
		bbox, conf = raw_detection[:4], raw_detection[4]
		if conf > DET_THR:
			#bbox = expand_bbox(bbox, img_w, img_h, k=0.1)
			cls_ = np.array([0.])
			detection = np.concatenate([bbox, conf[None], cls_])
			detections.append(detection)
	detections = np.stack(detections) if len(detections) > 0 else _empty_detections()
	num_heads = len(detections)
	if num_heads == 0:  # no head detected above DET_THR in this frame
		return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

	# 2. Detect & track head bboxes
	# if detections is just one, do not use tracker
	if detections.shape[0] == 1:
		head_bboxes = torch.from_numpy(detections[:, :4]).float()
		pids = torch.tensor([0], dtype=torch.int)
	else:
		# select the detection that has the highest confidence
		detections = detections[np.argsort(detections[:, 4])[::-1]][0]
		head_bboxes = torch.from_numpy(detections[:4][None, :]).float()
		pids = torch.tensor([0], dtype=torch.int)
		# tracks = tracker.update(detections, image_np)
		# if len(tracks) == 0: # sometimes tracker.update returns [] even when detections is not []
		# 	return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
		# pids = (tracks[:, 4] - 1).astype(int)
		# head_bboxes = torch.from_numpy(tracks[:, :4]).float()
	
	t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)
	
	# 3. Extract and transform heads
	heads = []
	for bbox in t_head_bboxes:
		head = TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224))
		heads.append(head)
	heads = torch.stack(heads)
	heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

	# 4. Transform Image
	image = TF.to_tensor(image)
	image = TF.resize(image, (224, 224))
	image = TF.normalize(image, mean=IMG_MEAN, std=IMG_STD)

	# 5. Normalize head bboxes
	scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	t_head_bboxes /= scale

	# 6. build input sample
	sample = {}
	sample["image"] = image.unsqueeze(0).to(DEVICE) # (1, 3, 224, 224)
	sample["heads"] = heads.unsqueeze(0).to(DEVICE) # (1, num_heads, 3, 224, 224)
	sample["head_bboxes"] = t_head_bboxes.unsqueeze(0).to(DEVICE) # (1, num_heads, 4)

	# 7. predict gaze
	with torch.no_grad():
		gaze_vecs, gaze_heatmaps, inouts = sharingan(sample)
		gaze_heatmaps = gaze_heatmaps.squeeze(0).cpu()
		gaze_vecs = gaze_vecs.squeeze(0).cpu()
		gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
		inouts = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()
  
	return gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids

def draw_gaze(
	image,
	head_bboxes,
	gaze_points,
	gaze_vecs,
	inouts,
	pids,
	gaze_heatmaps,
	heatmap_pid = None,
	frame_nb = None,
	colors = COLORS,
	filter_by_inout = False,
	alpha: float = 0.5,
	io_thr: float = 0.5, 
	gaze_pt_size: int = 10,
	gaze_vec_factor: float = 0.8,
	head_center_size: int = 10,
	thickness: int = 4,
	fs: float = 0.6,
):
	"""
	Draws gaze results on the given image.
 
	Args:
		image (np.ndarray): The input image on which to draw.
		head_bboxes (array-like): Bounding boxes for heads.
		gaze_points (array-like): Points representing gaze locations.
		gaze_vecs (array-like): Vectors representing gaze directions.
		inouts (array-like): In/out scores for each head.
		pids (array-like): Person IDs for each head.
		gaze_heatmaps (array-like): Heatmaps for gaze.
		heatmap_pid (int, optional): Person ID for which to draw the heatmap. Defaults to None.
		frame_nb (int, optional): Frame number to display on the image. Defaults to None.
		colors (array-like, optional): Colors to use for drawing. Defaults to COLORS.
		alpha (float, optional): Alpha blending value for heatmap overlay. Defaults to 0.5.
		io_thr (float, optional): Threshold for in/out scores to draw gaze points. Defaults to 0.5.
		gaze_pt_size (int, optional): Size of the gaze points. Defaults to 10.
		gaze_vec_factor (float, optional): Scaling factor for gaze vectors. Defaults to 0.8.
		head_center_size (int, optional): Size of the head center points. Defaults to 10.
		thickness (int, optional): Thickness of the drawing lines. Defaults to 4.
		fs (float, optional): Font scale for text. Defaults to 0.6.
	Returns:
		np.ndarray: The image with gaze results drawn on it.
	"""
	# Create canvas on which to draw predictions
	img_h, img_w, img_c = image.shape
	canvas = image.copy()
	
	# Scale of the drawing according to image resolution
	scale = max(img_h, img_w) / 1920
	fs *= scale
	thickness = int(scale * thickness)
	gaze_pt_size = int(scale * gaze_pt_size)
	head_center_size = int(scale * head_center_size)
	
	# Draw heatmap
	if heatmap_pid is not None:
		if len(gaze_heatmaps) == 0:
			raise ValueError("gaze_heatmaps must be provided if heatmap_pid is provided.")
		mask = (pids == heatmap_pid)
		if mask.sum() == 1: # only if detection found
			gaze_heatmap = gaze_heatmaps[mask]
			heatmap = TF.resize(gaze_heatmap, (img_h, img_w), antialias=True).squeeze().numpy()
			heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
			heatmap = cm.inferno(heatmap) * 255 
			canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)

			# Write pid being used for the heatmap
			hm_pid_text = f"Heatmap PID: {heatmap_pid}"
			(w_text, h_text), _ = cv2.getTextSize(hm_pid_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
			ul = (img_w - w_text - 20, img_h - h_text - 15)
			br = (img_w, img_h)
			cv2.rectangle(canvas, ul, br, (0, 0, 0), -1)
			hm_pid_text_loc = (img_w - w_text - 10, img_h - 10)
			cv2.putText(canvas, hm_pid_text, hm_pid_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)   

	# Draw head bboxes  
	if len(head_bboxes) > 0:
		if len(pids) == 0:
			raise ValueError("pids must be provided if head_bboxes is provided.")
			
		# Convert to numpy
		head_bboxes = head_bboxes.numpy() if isinstance(head_bboxes, torch.Tensor) else np.array(head_bboxes)
		inouts = inouts.numpy() if isinstance(inouts, torch.Tensor) else np.array(inouts)
		if head_bboxes.max() <= 1.0:
			head_bboxes = head_bboxes * np.array([img_w, img_h, img_w, img_h])
		head_bboxes = head_bboxes.astype(int)
		
		# Compute head center
		head_centers = np.hstack([(head_bboxes[:,[0]] + head_bboxes[:,[2]]) / 2, (head_bboxes[:,[1]] + head_bboxes[:,[3]]) / 2])
		head_centers = head_centers.astype(int)
		
		gaze_available = (len(gaze_points) > 0)
		if gaze_available and (len(inouts) == 0):
			raise ValueError("inouts must be provided if gaze_pts is provided.")
			
		if gaze_available:
			gaze_points = gaze_points.numpy() if isinstance(gaze_points, torch.Tensor) else np.array(gaze_points)
			if (gaze_points.max() <= 1.):
				gaze_points = gaze_points * np.array([img_w, img_h])
			gaze_points = gaze_points.astype(int)
			
		if gaze_vecs is not None:
			gaze_vecs = gaze_vecs.numpy() if isinstance(gaze_vecs, torch.Tensor) else np.array(gaze_vecs)
		
		for i, head_bbox in enumerate(head_bboxes):
			pid = pids[i]
			if (heatmap_pid is not None) and (heatmap_pid != pid):
				continue
			
			xmin, ymin, xmax, ymax = head_bbox
			head_radius = max(xmax-xmin, ymax-ymin) // 2
			color = colors[pid % len(colors)]
							
			# Compute Head Center
			head_center = head_centers[i]
		
			head_bbox_ul = (xmin, ymin)
			head_bbox_br = (xmax, ymax)
			head_center_ul = head_center - (head_center_size // 2)
			head_center_br = head_center + (head_center_size // 2)
			cv2.rectangle(canvas, head_center_ul, head_center_br, color, -1) # head center point
			# draw head bbox
			cv2.rectangle(canvas, head_bbox_ul, head_bbox_br, color, thickness) #
			# cv2.circle(canvas, head_center, head_radius, color, thickness) # head circle
			
			# Draw header
			io = inouts[i] if inouts is not None else "-"

			# draw io on top of the image
			# cv2.rectangle(canvas, (xmin, ymin - 20), (xmin + 80, ymin), color, -1)
			# cv2.putText(canvas, f"{io:.2f}", (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

			header_text = f"The gaze is IN frame with confidence: {io:.2f}"
			(w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
			
			# header_ul =  (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
			# header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
			header_ul =  (0, 0)
			header_br = (w_text + 10, h_text + 5)
			cv2.rectangle(canvas, header_ul, header_br, color, -1) # header bbox
			cv2.putText(canvas, header_text, (header_ul[0], int(h_text)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA) # header text
			
			if gaze_available and (io > io_thr or not filter_by_inout):
				gp = gaze_points[i]
				vec = (gp - head_center)
				vec = vec / (np.linalg.norm(vec) + 0.000001)
				intersection = head_center + (vec * head_radius).astype(int)
				cv2.line(canvas, intersection, gp, color, thickness)
				
				cv2.circle(canvas, gp, gaze_pt_size, color, -1)
				
			if gaze_vecs is not None:
				gv = gaze_vecs[i]
				cv2.arrowedLine(canvas, head_center, (head_center + gaze_vec_factor * head_radius * gv).astype(int), color, thickness)

			# add to the image the bounding box of the head
			# cv2.rectangle(canvas, head_bbox_ul, head_bbox_br, color, thickness)
				
				
	# Write frame number
	if frame_nb is not None:
		frame_nb = str(frame_nb)
		(w_text, h_text), _ = cv2.getTextSize(frame_nb, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
		nb_ul = (int((img_w - w_text) / 2), (img_h - h_text - 15))
		nb_br = (int((img_w + w_text) / 2), img_h)
		cv2.rectangle(canvas, nb_ul, nb_br, (0, 0, 0), -1)
		nb_text_loc = (int((img_w - w_text) / 2), (img_h - 10))
		cv2.putText(canvas, frame_nb, nb_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA) 

	return canvas


def main():
	demo_config = OmegaConf.load(args.config)
	head_detector_config = demo_config.get("head_detector", {})

	start = dt.datetime.now()
	
	# Path magic
	input_dir = Path(os.path.expanduser(args.input_dir))
	output_dir = Path(os.path.expanduser(args.output_dir))
	# input_dir = Path('/home/icub/lgori/video-testing-gaze/images/fruit-images/')
	# output_dir = Path('/home/icub/lgori/sharingan/video-maria')
	annotated_root = output_dir / "annotated"
	annotated_root.mkdir(parents=True, exist_ok=True)
	images = find_images(input_dir)
	if len(images) == 0:
		raise FileNotFoundError(f"No images found in: {input_dir}")
	print(colored(f"Processing {len(images)} images from {input_dir}", TERM_COLOR))

	# Load models
	tracker = load_tracker()
	head_detector = load_head_detector(head_detector_config, DEVICE)
	sharingan = load_sharingan_model(CKPT_PATH, DEVICE)
	print(colored(f"Loaded tracker, {head_detector_config.get('type', 'yolo')} head detector, and sharingan models.", TERM_COLOR))
	# Iterate over images and process
	num_frames_no_head = 0
	for image_path in tqdm(images, desc="Processing images"):
		rel_path = image_path.relative_to(input_dir)
		out_path = annotated_root / rel_path
		out_path.parent.mkdir(parents=True, exist_ok=True)

		image = Image.open(image_path).convert("RGB")
		image_np = np.array(image)

		# =============== Predict =============== #
		output = predict_gaze(image, sharingan, head_detector, tracker=tracker, frame_path=str(image_path))
		gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids = output
		if len(head_bboxes) == 0:  # no head detected: save the frame without annotations
			num_frames_no_head += 1
			Image.fromarray(image_np).save(out_path)
			continue

		# =============== Draw Prediction =============== #
		heatmap_pid = args.heatmap_pid if args.heatmap_pid >= 0 else None
		num_people = len(head_bboxes)
		pids = np.arange(num_people) if len(pids) == 0 else pids
		annotated = draw_gaze(image_np,
							 head_bboxes = head_bboxes,
							 gaze_points = gaze_points,
							 gaze_vecs = gaze_vecs if args.show_gaze_vec else None,
							 inouts = inouts,
							 pids = pids,
							 gaze_heatmaps = gaze_heatmaps,
							 heatmap_pid = heatmap_pid,
							 frame_nb = None,
							 colors = COLORS,
							 filter_by_inout = args.filter_by_inout,
							 alpha = 0.6,
							 gaze_pt_size = 20,
							 gaze_vec_factor = 0.6,
							 head_center_size = 18,
							 thickness = 10,
							 fs = 0.8,
							)

		Image.fromarray(annotated).save(out_path)
	
	end = dt.datetime.now()
	if num_frames_no_head > 0:
		print(colored(f"No head detected in {num_frames_no_head}/{len(images)} images (saved without annotations).", TERM_COLOR))
	print(colored(f"Finished. The script took {end - start}. Annotated images saved in: {annotated_root}", TERM_COLOR))
		

if __name__ == "__main__":
	main()
