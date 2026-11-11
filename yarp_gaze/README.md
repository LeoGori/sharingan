# Sharingan gaze YARP module

YARP module that runs the same single-person pipeline as `demo.py` on a stream of RGB images: head detection
(SAM3 or YOLOv5, set in `src/conf/demo.yaml`) → most confident head above the detection threshold → Sharingan.

- `sharingan_gaze_module.py`: the YARP module (`yarp.RFModule`).
- `gaze_estimator.py`: the inference pipeline, without YARP dependencies.

## Install

With [pixi](https://pixi.sh) (environment defined in `pixi.toml`, exact versions in `pixi.lock`):

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if pixi is not installed
cd yarp_gaze
pixi install                                   # creates the environment in yarp_gaze/.pixi
```

The environment includes YARP 3.12 (conda-forge), torch 2.3.1 with CUDA 12.1, and the SAM3 dependencies,
with versions matching the `sharingan` training environment.

Alternatively, in the `sharingan` conda environment (pin numpy: numpy 2 breaks the installed torch 2.3.1):

```bash
mamba install -n sharingan -c conda-forge yarp-python "numpy<2"
```

## Run

```bash
cd yarp_gaze
pixi run yarpserver &            # if not already running on the network
pixi run start --name sharinganGaze
```

or, without pixi: `python yarp_gaze/sharingan_gaze_module.py --name sharinganGaze`.

Options (all optional):

| Option | Default | Description |
|---|---|---|
| `--name` | `sharinganGaze` | Prefix of the port names |
| `--ckpt` | `checkpoints/gf+vat+ours-head-bg-swap-mg0.ckpt` | Sharingan checkpoint |
| `--config` | `src/conf/demo.yaml` | Head detector configuration |
| `--det_thr` | `0.4` | Head detection threshold |
| `--device` | `cuda` if available, else `cpu` | Torch device |
| `--period` | `0.01` | Module period in seconds (polling of the input port) |
| `--normalized` | off | Publish the gaze point in [0, 1] instead of pixels |

## Ports

| Port | Type | Content |
|---|---|---|
| `/sharinganGaze/image:i` | `ImageOf<PixelRgb>` | Input RGB image |
| `/sharinganGaze/gaze:o` | `Bottle` | `gaze_x gaze_y inout` (3 float64) |
| `/sharinganGaze/outputImage:o` | `ImageOf<PixelRgb>` | Input image with the detected head box (green, with detection score), the gaze point (red dot) and the in-out score written in a banner on top |
| `/sharinganGaze/rpc` | rpc | `quit` closes the module |

- `gaze_x gaze_y`: gaze point in pixels of the input image (or in [0, 1] with `--normalized`).
- `inout`: probability in [0, 1] that the gaze target is inside the image.
- When no head is detected, the module publishes `-1 -1 0`.
- The output carries the envelope (timestamp) of the input image it was computed from.
- The module always processes the most recent image, so frames are dropped if the input is faster than inference.

Example:

```bash
yarp connect /<camera>/rgbImage:o /sharinganGaze/image:i
yarp read ... /sharinganGaze/gaze:o
```
