#
# YARP module running Sharingan gaze estimation on a stream of RGB images.
#
# Ports (with the default --name sharinganGaze):
#   /sharinganGaze/image:i  (input,  yarp::sig::ImageOf<PixelRgb>)
#   /sharinganGaze/gaze:o   (output, Bottle with 3 float64: gaze_x gaze_y inout)
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

import numpy as np
import yarp
from termcolor import colored

from gaze_estimator import DEFAULT_CKPT_PATH, DEFAULT_CONFIG_PATH, DET_THR, TERM_COLOR, SharinganGazeEstimator

NO_HEAD_OUTPUT = (-1.0, -1.0, 0.0)


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
		self.rpc_port = yarp.Port()
		for port, port_name in ((self.image_port, "image:i"), (self.gaze_port, "gaze:o"), (self.rpc_port, "rpc")):
			if not port.open(f"/{name}/{port_name}"):
				print(colored(f"Cannot open port /{name}/{port_name}", "red"))
				return False
		self.attach(self.rpc_port)

		# Buffer receiving the YARP image as a numpy array (reallocated when the image size changes)
		self.image_buffer = None
		self.yarp_image_buffer = yarp.ImageRgb()

		print(colored(f"Module /{name} ready: image -> /{name}/image:i, gaze (x, y, inout) -> /{name}/gaze:o", TERM_COLOR))
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
		if result is None:
			gaze_x, gaze_y, inout = NO_HEAD_OUTPUT
		else:
			gaze_x, gaze_y, inout = result
			if not self.normalized:
				gaze_x, gaze_y = gaze_x * image.shape[1], gaze_y * image.shape[0]

		bottle = self.gaze_port.prepare()
		bottle.clear()
		bottle.addFloat64(gaze_x)
		bottle.addFloat64(gaze_y)
		bottle.addFloat64(inout)
		self.gaze_port.setEnvelope(envelope)
		self.gaze_port.write()
		return True

	def interruptModule(self):
		for port in (self.image_port, self.gaze_port, self.rpc_port):
			port.interrupt()
		return True

	def close(self):
		for port in (self.image_port, self.gaze_port, self.rpc_port):
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
