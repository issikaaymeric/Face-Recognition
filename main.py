"""
Controls:
    Q / ESC  -  quit
    S        -  save a screenshot
"""

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


#  Configurations

@dataclass
class Config:
    ref_image_path: str = "elon.jpg"
    camera_index:   int = 0
    frame_width:    int = 640
    frame_height:   int = 480
    check_interval: int = 30          # run DeepFace every N frames
    model_name:     str = "VGG-Face"  # Facenet, ArcFace, etc.
    distance_metric:str = "cosine"    # cosine | euclidean | euclidean_l2
    confidence_history: int = 5       # frames to smooth confidence display


#  Shared state (thread-safe) 

@dataclass
class VerifyState:
    matched:    bool  = False
    distance:   float = 1.0           # lower = more similar
    threshold:  float = 0.40          # default cosine threshold
    processing: bool  = False
    last_check: float = field(default_factory=time.time)
    history:    list  = field(default_factory=list)   # recent distances
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, verified: bool, distance: float, threshold: float):
        with self.lock:
            self.matched   = verified
            self.distance  = distance
            self.threshold = threshold
            self.history.append(distance)
            if len(self.history) > 10:
                self.history.pop(0)
            self.processing = False
            self.last_check = time.time()

    def fail(self):
        with self.lock:
            self.matched    = False
            self.processing = False

    def snapshot(self):
        with self.lock:
            return self.matched, self.distance, self.threshold, self.processing, list(self.history)


#  Worker thread 

def verify_worker(frame: np.ndarray, ref_img: np.ndarray,
                  state: VerifyState, cfg: Config):
    try:
        result = DeepFace.verify(
            img1_path        = frame,
            img2_path        = ref_img.copy(),
            model_name       = cfg.model_name,
            distance_metric  = cfg.distance_metric,
            enforce_detection= False,
        )
        state.update(
            verified  = result["verified"],
            distance  = result["distance"],
            threshold = result["threshold"],
        )
    except Exception as exc:
        print(f"[DeepFace] Error: {exc}")
        state.fail()


# Drawing helpers

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (50,  220,  80)
RED    = (50,   60, 220)
YELLOW = (30,  200, 230)
WHITE  = (240, 240, 240)
BLACK  = (10,   10,  10)
GREY   = (140, 140, 140)


def draw_overlay(frame: np.ndarray, state: VerifyState, fps: float, frame_no: int):
    """Render all UI elements on the frame."""
    matched, distance, threshold, processing, history = state.snapshot()
    h, w = frame.shape[:2]

    #  Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), BLACK, -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "FACE VERIFY", (12, 36), FONT, 1.1, WHITE, 2, cv2.LINE_AA)

    # FPS counter
    fps_txt = f"FPS {fps:05.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, FONT_SMALL, 0.55, 1)
    cv2.putText(frame, fps_txt, (w - tw - 12, 32), FONT_SMALL, 0.55, GREY, 1, cv2.LINE_AA)

    #  Bottom status bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 62), (w, h), BLACK, -1)
    cv2.addWeighted(bar2, 0.60, frame, 0.40, 0, frame)

    if processing:
        color, label = YELLOW, "ANALYZING…"
        blink = int(time.time() * 3) % 2 == 0
        if blink:
            cv2.putText(frame, label, (16, h - 22), FONT, 1.0, color, 2, cv2.LINE_AA)
    else:
        color = GREEN if matched else RED
        label = "✓  MATCH" if matched else "✗  NO MATCH"
        cv2.putText(frame, label, (16, h - 22), FONT, 1.0, color, 2, cv2.LINE_AA)

    #  Distance / confidence meter 
    conf = max(0.0, 1.0 - distance / max(threshold, 1e-6))   # 0..1+
    conf_clamped = min(conf, 1.0)
    bar_w = int((w // 2 - 24) * conf_clamped)
    bar_x = w // 2 + 12
    bar_y = h - 42
    bar_h = 14

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w // 2 - 24, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w // 2 - 24, bar_y + bar_h), GREY, 1)

    cv2.putText(frame, f"Conf {conf_clamped*100:4.1f}%  dist {distance:.3f}/{threshold:.3f}",
                (bar_x, h - 18), FONT_SMALL, 0.44, GREY, 1, cv2.LINE_AA)

    #  Sparkline (distance history) 
    if len(history) >= 2:
        pts_x = np.linspace(w - 110, w - 14, len(history)).astype(int)
        pts_y = np.interp(history, [0, 1], [h - 64, h - 14 - bar_h - 10]).astype(int)
        for i in range(1, len(pts_x)):
            cv2.line(frame, (pts_x[i-1], pts_y[i-1]), (pts_x[i], pts_y[i]), YELLOW, 1, cv2.LINE_AA)

    # Corner frame decorations
    corner_len = 28
    thickness  = 2
    corners = [(0, 0, 1, 1), (w - corner_len, 0, -1, 1),
               (0, h - corner_len, 1, -1), (w - corner_len, h - corner_len, -1, -1)]
    for cx, cy, sx, sy in corners:
        cv2.line(frame, (cx, cy), (cx + corner_len * sx, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + corner_len * sy), color, thickness)

    return frame


# Main loop 

def run(cfg: Config):
    #  Load reference image 
    ref_path = Path(cfg.ref_image_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise ValueError(f"Could not read image: {ref_path}")
    print(f"[+] Reference loaded: {ref_path.name}  ({ref_img.shape[1]}×{ref_img.shape[0]})")

    #  Open camera
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.camera_index}")
    print(f"[+] Camera opened  ({cfg.frame_width}×{cfg.frame_height})")

    state    = VerifyState()
    frame_no = 0
    prev_t   = time.time()
    fps      = 0.0
    screenshot_dir = Path("screenshots")

    print("[+] Running — press Q/ESC to quit, S to screenshot\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame capture failed — exiting.")
                break

            #  FPS 
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            #  Trigger verify thread 
            if frame_no % cfg.check_interval == 0:
                with state.lock:
                    already = state.processing
                if not already:
                    with state.lock:
                        state.processing = True
                    t = threading.Thread(
                        target=verify_worker,
                        args=(frame.copy(), ref_img, state, cfg),
                        daemon=True,
                    )
                    t.start()

            #  Render UI
            draw_overlay(frame, state, fps, frame_no)
            cv2.imshow("Live Face Verification", frame)

            #  Key handling 
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # Q or ESC
                break
            elif key == ord("s"):              # screenshot
                screenshot_dir.mkdir(exist_ok=True)
                ts  = time.strftime("%Y%m%d_%H%M%S")
                out = screenshot_dir / f"snap_{ts}.png"
                cv2.imwrite(str(out), frame)
                print(f"[+] Screenshot saved → {out}")

            frame_no += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Exited cleanly.")


#  Entry point

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Real-time face verification with DeepFace")
    p.add_argument("--ref",      default="elon.jpg",   help="Path to reference image")
    p.add_argument("--cam",      type=int, default=0,  help="Camera index (default 0)")
    p.add_argument("--interval", type=int, default=30, help="Verify every N frames")
    p.add_argument("--model",    default="VGG-Face",   help="DeepFace model name")
    a = p.parse_args()
    return Config(
        ref_image_path = a.ref,
        camera_index   = a.cam,
        check_interval = a.interval,
        model_name     = a.model,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)