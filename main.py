"""Tkinter GUI: live camera plate detection + upload-image fallback.

Designed for Raspberry Pi 4 (runs on any machine with a webcam though).

Usage:
    python app/main.py \\
        --ocr-model runs/cambodia_v1/2026-04-17_08-43-49/best.onnx \\
        --plate-config configs/cambodia_plate_config.yaml
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from tkinter import Tk, Frame, Button, Label, filedialog, StringVar, BOTH, LEFT, RIGHT, TOP, X, Y, NW

import cv2
from PIL import Image, ImageTk

try:
    from app.pipeline import PlatePipeline, PlateResult  # running as: python -m app.main
except ModuleNotFoundError:
    from pipeline import PlatePipeline, PlateResult  # running as: python main.py


PREVIEW_W, PREVIEW_H = 800, 480  # fits the official 7" Pi screen


def parse_args():
    APP_DIR = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-model", type=Path,
                    default=APP_DIR / "best.onnx")
    ap.add_argument("--plate-config", type=Path,
                    default=APP_DIR / "cambodia_plate_config.yaml")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (OpenCV index, or picamera2 camera num).")
    ap.add_argument("--camera-backend", choices=["auto", "opencv", "picamera2"], default="auto",
                    help="auto: try picamera2 first on Pi, else OpenCV. "
                         "picamera2: Pi CSI/libcamera. opencv: USB webcam or V4L2 device.")
    ap.add_argument("--detect-every", type=int, default=5,
                    help="Run detection every N frames to keep UI smooth on Pi.")
    ap.add_argument("--min-det-conf", type=float, default=0.35)
    return ap.parse_args()


class _BaseCameraWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_q: Queue = Queue(maxsize=1)
        self._stop = threading.Event()

    def _push(self, frame) -> None:
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
            except Empty:
                pass
        self.frame_q.put(frame)

    def stop(self) -> None:
        self._stop.set()


class OpenCVCameraWorker(_BaseCameraWorker):
    """USB webcam / V4L2 capture via OpenCV."""
    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index

    def _open_capture(self):
        # On Raspberry Pi OS, forcing V4L2 avoids some blank-frame cases
        # seen with OpenCV's generic backend auto-selection.
        candidates = []
        if hasattr(cv2, "CAP_V4L2"):
            candidates.append(("V4L2", cv2.CAP_V4L2))
        candidates.append(("AUTO", None))

        for name, backend in candidates:
            cap = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[camera] OpenCV backend={name}, index={self.camera_index}")
            return cap
        return None

    def run(self) -> None:
        cap = self._open_capture()
        if cap is None:
            print(f"[camera] OpenCV could not open index {self.camera_index}")
            return
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self._push(frame)
        cap.release()


class Picamera2CameraWorker(_BaseCameraWorker):
    """Raspberry Pi CSI camera via libcamera / picamera2. Produces BGR frames."""
    def __init__(self, camera_num: int = 0, size: tuple[int, int] = (640, 480)):
        super().__init__()
        self.camera_num = camera_num
        self.size = size

    def run(self) -> None:
        try:
            from picamera2 import Picamera2  # system pkg on Pi: python3-picamera2
        except ImportError as e:
            print(f"[camera] picamera2 not importable: {e}\n"
                  "        Install: sudo apt install -y python3-picamera2\n"
                  "        Then recreate the venv with --system-site-packages.")
            return

        try:
            picam2 = Picamera2(camera_num=self.camera_num)
        except Exception as e:  # noqa: BLE001
            print(f"[camera] Picamera2 open failed: {e}")
            return

        # 'RGB888' in picamera2 is actually packed BGR in the numpy array —
        # matches what OpenCV + our pipeline expect.
        config = picam2.create_video_configuration(main={"size": self.size, "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        try:
            while not self._stop.is_set():
                frame = picam2.capture_array()  # HxWx3 BGR
                self._push(frame)
        finally:
            picam2.stop()
            picam2.close()


def build_camera_worker(backend: str, camera_index: int) -> _BaseCameraWorker:
    if backend == "opencv":
        return OpenCVCameraWorker(camera_index)
    if backend == "picamera2":
        return Picamera2CameraWorker(camera_num=camera_index)
    # auto: prefer picamera2 if it imports (we're on a Pi with the CSI stack)
    try:
        import picamera2  # noqa: F401
        print("[camera] auto-selected picamera2 backend.")
        return Picamera2CameraWorker(camera_num=camera_index)
    except ImportError:
        print("[camera] auto-selected OpenCV backend.")
        return OpenCVCameraWorker(camera_index)


class App:
    def __init__(
        self,
        root: Tk,
        pipeline: PlatePipeline,
        camera_index: int,
        camera_backend: str,
        detect_every: int,
    ):
        self.root = root
        self.pipeline = pipeline
        self.detect_every = detect_every

        self.root.title("Cambodia Plate OCR")
        self.root.configure(bg="#1e1e1e")

        top = Frame(root, bg="#1e1e1e")
        top.pack(side=TOP, fill=BOTH, expand=True)

        self.video_label = Label(top, bg="#000000", width=PREVIEW_W, height=PREVIEW_H)
        self.video_label.pack(side=LEFT, padx=8, pady=8)

        right = Frame(top, bg="#1e1e1e")
        right.pack(side=RIGHT, fill=Y, padx=8, pady=8)

        Label(right, text="Detected plate", fg="#bbbbbb", bg="#1e1e1e",
              font=("Helvetica", 12)).pack(anchor=NW)
        self.plate_var = StringVar(value="—")
        Label(right, textvariable=self.plate_var, fg="#00ff88", bg="#1e1e1e",
              font=("Helvetica", 28, "bold")).pack(anchor=NW, pady=(0, 12))

        self.info_var = StringVar(value="Starting camera…")
        Label(right, textvariable=self.info_var, fg="#cccccc", bg="#1e1e1e",
              font=("Helvetica", 11), justify=LEFT, wraplength=260).pack(anchor=NW, pady=(0, 16))

        Button(right, text="Upload image", command=self.on_upload,
               bg="#0078d7", fg="white", font=("Helvetica", 12, "bold"),
               relief="flat", padx=12, pady=8).pack(fill=X, pady=4)
        Button(right, text="Pause / Resume", command=self.toggle_pause,
               bg="#555555", fg="white", font=("Helvetica", 12, "bold"),
               relief="flat", padx=12, pady=8).pack(fill=X, pady=4)
        Button(right, text="Quit", command=self.shutdown,
               bg="#a33", fg="white", font=("Helvetica", 12, "bold"),
               relief="flat", padx=12, pady=8).pack(fill=X, pady=4)

        self.paused = False
        self.frame_count = 0
        self.no_frame_count = 0
        self.last_results: list[PlateResult] = []
        self.mode = "live"  # or "still"
        self.still_image = None

        self.camera = build_camera_worker(camera_backend, camera_index)
        self.camera.start()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.after(30, self.tick)

    def toggle_pause(self) -> None:
        # From a still upload, resume = return to live camera.
        if self.mode == "still":
            self.mode = "live"
            self.still_image = None
            return
        self.paused = not self.paused

    def on_upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.info_var.set(f"Could not read {path}")
            return
        results = self.pipeline.run(img)
        self.last_results = results
        self.still_image = img
        self.mode = "still"
        self._render(img, results)
        self._update_text(results, source=Path(path).name)

    def tick(self) -> None:
        if self.mode == "still":
            # Keep showing the uploaded image until user resumes camera
            self.root.after(100, self.tick)
            return

        if self.paused:
            self.root.after(100, self.tick)
            return

        try:
            frame = self.camera.frame_q.get_nowait()
        except Empty:
            self.no_frame_count += 1
            if self.no_frame_count == 100:
                self.info_var.set(
                    "No camera frames yet. If this is a Pi Camera, try "
                    "--camera-backend picamera2 and install python3-picamera2."
                )
            self.root.after(20, self.tick)
            return
        self.no_frame_count = 0

        self.frame_count += 1
        if self.frame_count % self.detect_every == 0:
            try:
                self.last_results = self.pipeline.run(frame)
                self._update_text(self.last_results, source="camera")
            except Exception as e:  # noqa: BLE001
                self.info_var.set(f"Pipeline error: {e}")

        self._render(frame, self.last_results)
        self.root.after(15, self.tick)

    def _render(self, bgr, results) -> None:
        annotated = PlatePipeline.draw(bgr, results)
        h, w = annotated.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        new_size = (int(w * scale), int(h * scale))
        annotated = cv2.resize(annotated, new_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=img, width=new_size[0], height=new_size[1])
        self.video_label.image = img  # keep ref

    def _update_text(self, results, source: str) -> None:
        if not results:
            self.plate_var.set("—")
            self.info_var.set(f"{source}: no plate detected")
            return
        top = max(results, key=lambda r: r.det_confidence)
        self.plate_var.set(top.text or "(unreadable)")
        lines = [f"{source}: {len(results)} plate(s)"]
        for r in results[:4]:
            lines.append(f"  {r.text or '??'}  det={r.det_confidence:.2f}  ocr={r.confidence:.2f}")
        self.info_var.set("\n".join(lines))
        # Tapping upload then clicking preview returns to live
        if self.mode == "still":
            lines.append("\nClick 'Pause / Resume' to return to live camera.")
            self.info_var.set("\n".join(lines))

    def shutdown(self) -> None:
        self.camera.stop()
        self.root.after(100, self.root.destroy)


def main() -> int:
    args = parse_args()
    if not args.ocr_model.exists():
        raise SystemExit(f"OCR model not found: {args.ocr_model}")
    if not args.plate_config.exists():
        raise SystemExit(f"Plate config not found: {args.plate_config}")

    pipeline = PlatePipeline(
        ocr_onnx_path=args.ocr_model,
        plate_config_path=args.plate_config,
        min_det_conf=args.min_det_conf,
    )

    root = Tk()
    App(root, pipeline, args.camera, args.camera_backend, args.detect_every)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
