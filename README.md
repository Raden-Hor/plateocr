# Plate OCR Desktop/Pi App

Small Tkinter GUI that runs the Cambodia OCR model on a live camera feed or an uploaded image. Uses ONNX so it stays lightweight enough for a Raspberry Pi 4.

Two stages:
1. **Detector** — `open-image-models` YOLOv9 ONNX finds plate boxes.
2. **Recognizer** — your fine-tuned `best.onnx` reads the text in each box.

## What to copy to the Pi

The entire `app/` folder is self-contained — it already has the model (`best.onnx`) and plate config (`cambodia_plate_config.yaml`) bundled inside. Just copy it:

```bash
scp -r app/ pi@raspberrypi.local:~/plate-ocr/
```

Resulting layout on the Pi:
```
~/plate-ocr/
  main.py
  pipeline.py
  best.onnx
  cambodia_plate_config.yaml
  requirements.txt
  README.md
```

You do **not** need `dataset/`, `models/`, `configs/`, `runs/`, `*.keras`, or any training scripts.

## Pi 4 setup (Raspberry Pi OS, 64-bit Bookworm recommended)

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk libatlas-base-dev libjpeg-dev libopenjp2-7 \
                    libtiff6 libavcodec-dev libavformat-dev libswscale-dev \
                    v4l-utils

cd ~/plate-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Camera

- **USB webcam**: works out of the box at `--camera 0`.
- **Pi Camera Module (libcamera/CSI)**: enable with `sudo raspi-config` → Interface → Camera. On Bookworm, `libcamera` exposes it as a V4L2 device (`/dev/video0`) so OpenCV picks it up via index 0.
- If preview is black/blank on Pi Camera, force the Pi backend:
  `python main.py --camera-backend picamera2 --camera 0`

List cameras: `v4l2-ctl --list-devices`.

## Run

**On the Pi** (from inside `~/plate-ocr/`):

```bash
source .venv/bin/activate
python main.py
```

Model + config default to the files sitting next to `main.py`, so no flags needed. To override:

```bash
python main.py --camera 1 --detect-every 8
```

**On your development Mac** (from the repo root — uses `app/` as a package):

```bash
source .venv/bin/activate
python -m app.main
```

> **Homebrew Python 3.11 on macOS** ships without Tk bindings and fails with
> `ModuleNotFoundError: No module named '_tkinter'`. Install them once:
> ```bash
> brew install python-tk@3.11
> ```
> No `pip` install needed — it lands in the system Python 3.11 and the venv picks it up automatically.

## Controls

- **Upload image** — pick a JPG/PNG from disk; plates are annotated on the preview and the text shows on the right panel.
- **Pause / Resume** — freezes the live feed. If you uploaded an image, this button returns to the live camera.
- **Quit** — releases the camera and closes the window.

## Tuning for Pi 4 performance

- `--detect-every 5` (default): run detection on every 5th frame. Increase to 8–10 if the UI feels sluggish.
- The detector default is `yolo-v9-t-384-license-plate-end2end` (the small/fast model). Edit `app/pipeline.py` to use a larger model if you need better recall at the cost of FPS.
- Expected: ~3–6 FPS end-to-end on a Pi 4 8GB with a USB webcam at 640x480.

## Run on boot (optional)

Create `/etc/systemd/system/plate-ocr.service`:

```ini
[Unit]
Description=Plate OCR GUI
After=graphical.target

[Service]
User=pi
Environment=DISPLAY=:0
WorkingDirectory=/home/pi/plate-ocr
ExecStart=/home/pi/plate-ocr/.venv/bin/python -m app.main
Restart=on-failure

[Install]
WantedBy=graphical.target
```

Then `sudo systemctl enable --now plate-ocr`.

## Troubleshooting

- **`could not open index 0`** — no camera found. Try `--camera 1`, or check `v4l2-ctl --list-devices`.
- **Black/blank live preview on Pi Camera** — run with:
  `python main.py --camera-backend picamera2 --camera 0`
  and install `picamera2` system package:
  `sudo apt install -y python3-picamera2`
- **`ImportError: libGL.so.1`** — install `libgl1`: `sudo apt install -y libgl1`.
- **OCR prints garbage text** — your `best.onnx` or `cambodia_plate_config.yaml` doesn't match. Re-copy both from the training machine together (they're a pair).
- **`ModuleNotFoundError: No module named '_tkinter'`** — Tk isn't installed for your Python. On the Pi/Ubuntu: `sudo apt install -y python3-tk`. On macOS + Homebrew Python 3.11: `brew install python-tk@3.11`.
- **Slow first frame** — onnxruntime warms up on the first inference; expect a 1–2s pause.
