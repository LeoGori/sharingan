#
# YARP module running Sharingan gaze estimation on a stream of RGB images.
#
# Ports (with the default --name sharinganGaze):
#   /sharinganGaze/image:i  (input,  yarp::sig::ImageOf<PixelRgb>)
#   /sharinganGaze/gaze:o   (output, Bottle with 3 float64: gaze_x gaze_y inout)
#   /sharinganGaze/outputImage:o (output, yarp::sig::ImageOf<PixelRgb>: input image with the detected head box,
#                                 the gaze point and the inout score drawn on it)
#   /sharinganGaze/rpc      (rpc, send "quit" to close the module)
#
# The gaze point is in pixels of the input image (or in [0, 1] with --normalized), and inout is the
# probability in [0, 1] that the gaze target is inside the image. When no head is detected, the module
# publishes "-1 -1 0". The output carries the envelope (timestamp) of the input image it was computed from.
#
# Usage:
#   python yarp_gaze/sharingan_gaze_module.py [--name sharinganGaze] [--ckpt <path>] [--config <demo.yaml>]
#                                             [--det_thr 0.4] [--device cuda] [--period 0.01] [--normalized]
#

import sys

import cv2
import numpy as np
import yarp
from termcolor import colored

from gaze_estimator import DEFAULT_CKPT_PATH, DEFAULT_CONFIG_PATH, DET_THR, TERM_COLOR, SharinganGazeEstimator

NO_HEAD_OUTPUT = (-1.0, -1.0, 0.0)
GAZE_COLOR = (255, 0, 0)  # RGB
HEAD_COLOR = (0, 255, 0)  # RGB


def draw_gaze(image, gaze_px, inout, head_bbox=None, head_conf=None):
	"""Returns a copy of the RGB image with the head box, the gaze point and the inout score written on top."""
	image = image.copy()
	h, w = image.shape[:2]
	scale = max(0.5, min(w, h) / 640)
	thickness = max(1, round(2 * scale))
	font = cv2.FONT_HERSHEY_SIMPLEX

	if head_bbox is not None:
		x1, y1, x2, y2 = (round(v) for v in head_bbox)
		cv2.rectangle(image, (x1, y1), (x2, y2), HEAD_COLOR, thickness, cv2.LINE_AA)
		label = f"head {head_conf:.2f}"
		label_scale = 0.5 * scale
		(label_w, label_h), label_base = cv2.getTextSize(label, font, label_scale, thickness)
		label_y = y1 - label_base - 2 if y1 - label_h - label_base - 2 >= 0 else y2 + label_h + 2
		cv2.putText(image, label, (x1, label_y), font, label_scale, HEAD_COLOR, thickness, cv2.LINE_AA)

	if gaze_px is None:
		text = "No head detected"
	else:
		x, y = round(gaze_px[0]), round(gaze_px[1])
		radius = max(4, round(8 * scale))
		cv2.circle(image, (x, y), radius, GAZE_COLOR, -1, cv2.LINE_AA)
		cv2.circle(image, (x, y), radius + thickness, (255, 255, 255), thickness, cv2.LINE_AA)
		text = f"Gaze point: ({x}, {y}) px | In-out score (gaze target inside image): {inout:.2f}"

	# Text on a dark banner at the top of the image, shrunk to fit the image width
	font_scale = 0.6 * scale
	(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
	if text_w > w - 10:
		font_scale *= (w - 10) / text_w
		(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
	banner_h = text_h + baseline + 10
	image[:banner_h] = (image[:banner_h] * 0.4).astype(np.uint8)
	cv2.putText(image, text, (5, 5 + text_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
	return image


class SharinganGazeModule(yarp.RFModule):
	def configure(self, rf):
		name = rf.check("name", yarp.Value("sharinganGaze")).asString()
		ckpt_path = rf.check("ckpt", yarp.Value(str(DEFAULT_CKPT_PATH))).asString()
		config_path = rf.check("config", yarp.Value(str(DEFAULT_CONFIG_PATH))).asString()
		det_thr = rf.check("det_thr", yarp.Value(DET_THR)).asFloat64()
		device = rf.check("device", yarp.Value("")).asString()
		self.period = rf.check("period", yarp.Value(0.01)).asFloat64()
		self.normalized = rf.check("normalized")

		# Load models before opening the ports, so that clients connect to a ready module
		self.estimator = SharinganGazeEstimator(ckpt_path, config_path, det_thr=det_thr, device=device or None)

		self.image_port = yarp.BufferedPortImageRgb()
		self.gaze_port = yarp.BufferedPortBottle()
		self.output_image_port = yarp.BufferedPortImageRgb()
		self.rpc_port = yarp.Port()
		self.ports = (self.image_port, self.gaze_port, self.output_image_port, self.rpc_port)
		port_names = ("image:i", "gaze:o", "outputImage:o", "rpc")
		for port, port_name in zip(self.ports, port_names):
			if not port.open(f"/{name}/{port_name}"):
				print(colored(f"Cannot open port /{name}/{port_name}", "red"))
				return False
		self.attach(self.rpc_port)

		# Buffer receiving the YARP image as a numpy array (reallocated when the image size changes)
		self.image_buffer = None
		self.yarp_image_buffer = yarp.ImageRgb()
		self.yarp_output_buffer = yarp.ImageRgb()

		print(colored(
			f"Module /{name} ready: image -> /{name}/image:i, gaze (x, y, inout) -> /{name}/gaze:o, "
			f"annotated image -> /{name}/outputImage:o",
			TERM_COLOR,
		))
		return True

	def getPeriod(self):
		return self.period

	def _yarp_to_numpy(self, yarp_image):
		width, height = yarp_image.width(), yarp_image.height()
		if self.image_buffer is None or self.image_buffer.shape[:2] != (height, width):
			self.image_buffer = np.zeros((height, width, 3), dtype=np.uint8)
			self.yarp_image_buffer.resize(width, height)
			self.yarp_image_buffer.setExternal(self.image_buffer, width, height)
		self.yarp_image_buffer.copy(yarp_image)
		return self.image_buffer

	def updateModule(self):
		yarp_image = self.image_port.read(False)  # non-blocking: process the latest image only
		if yarp_image is None:
			return True

		envelope = yarp.Stamp()
		self.image_port.getEnvelope(envelope)
		image = self._yarp_to_numpy(yarp_image)

		result = self.estimator.predict(image)
		gaze_px, head_bbox, head_conf = None, None, None
		if result is None:
			gaze_x, gaze_y, inout = NO_HEAD_OUTPUT
		else:
			gaze_x, gaze_y, inout, head_bbox, head_conf = result
			gaze_px = (gaze_x * image.shape[1], gaze_y * image.shape[0])
			if not self.normalized:
				gaze_x, gaze_y = gaze_px

		bottle = self.gaze_port.prepare()
		bottle.clear()
		bottle.addFloat64(gaze_x)
		bottle.addFloat64(gaze_y)
		bottle.addFloat64(inout)
		self.gaze_port.setEnvelope(envelope)
		self.gaze_port.write()

		if self.output_image_port.getOutputCount() > 0:  # draw only when someone is reading
			self._publish_image(draw_gaze(image, gaze_px, inout, head_bbox, head_conf), envelope)
		return True

	def _publish_image(self, image, envelope):
		height, width = image.shape[:2]
		self.yarp_output_buffer.resize(width, height)
		self.yarp_output_buffer.setExternal(image, width, height)
		output = self.output_image_port.prepare()
		output.copy(self.yarp_output_buffer)
		self.output_image_port.setEnvelope(envelope)
		self.output_image_port.write()

	def interruptModule(self):
		for port in self.ports:
			port.interrupt()
		return True

	def close(self):
		for port in self.ports:
			port.close()
		return True


def main():
	yarp.Network.init()
	if not yarp.Network.checkNetwork():
		print(colored("YARP server not available (run `yarpserver` first).", "red"))
		return 1

	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)
	module = SharinganGazeModule()
	ret = module.runModule(rf)
	yarp.Network.fini()
	return ret


if __name__ == "__main__":
	sys.exit(main())
