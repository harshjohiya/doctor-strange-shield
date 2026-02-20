"""
main.py
-------
Doctor Strange Magic Shield – webcam application entry point.

Controls
~~~~~~~~
    q  –  quit
    s  –  toggle shield on / off

Usage
~~~~~
    python main.py

Requirements
~~~~~~~~~~~~
    pip install opencv-python mediapipe numpy
"""

import sys
import time

import cv2
import numpy as np

from hand_tracking  import HandTracker
from shield_renderer import draw_shield, GOLD_PRIMARY

# ---------------------------------------------------------------------------
# Configuration constants  (tweak here without touching logic below)
# ---------------------------------------------------------------------------
CAM_INDEX          = 0       # 0 = default webcam; change for external cameras
CAM_WIDTH          = 1280    # requested capture width  (px)
CAM_HEIGHT         = 720     # requested capture height (px)

ANGLE_SPEED        = 1.6     # degrees added to the global rotation per frame
                             # – raise for faster spin, lower for slower

MAX_HANDS          = 2       # detect up to 2 hands simultaneously
DETECT_CONFIDENCE  = 0.70    # MediaPipe initial-detection threshold
TRACK_CONFIDENCE   = 0.60    # MediaPipe tracking threshold

SHIELD_COLOR       = GOLD_PRIMARY   # can be swapped to any BGR tuple

# UI text styling
UI_FONT       = cv2.FONT_HERSHEY_SIMPLEX
UI_COLOR      = (0, 200, 255)   # cyan-gold
UI_SCALE      = 0.72
UI_THICKNESS  = 2


# ---------------------------------------------------------------------------
# Helper: overlay FPS counter and status text
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    fps: float,
    num_hands: int,
    shield_on: bool,
    palm_open_flags: list,
) -> None:
    """Render a minimal heads-up display in the top-left corner."""
    shield_status = "ON  [s]" if shield_on else "OFF [s]"

    lines = [
        f"FPS     : {fps:5.1f}",
        f"Shield  : {shield_status}",
        f"Hands   : {num_hands}",
    ]
    # Append per-hand open/closed state
    for i, is_open in enumerate(palm_open_flags, start=1):
        pose = "open" if is_open else "fist"
        lines.append(f"  Hand {i} : {pose}")

    lines.append("[q] quit")

    for row, text in enumerate(lines):
        y = 30 + row * 26
        cv2.putText(frame, text, (10, y), UI_FONT, UI_SCALE,
                    UI_COLOR, UI_THICKNESS, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Camera setup ------------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAM_INDEX}.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened at {actual_w}×{actual_h}")

    # ---- Hand tracker (reused every frame – MediaPipe graph stays alive) ---
    tracker = HandTracker(
        max_hands=MAX_HANDS,
        detection_confidence=DETECT_CONFIDENCE,
        tracking_confidence=TRACK_CONFIDENCE,
    )

    # ---- Application state -------------------------------------------------
    angle        = 0.0    # global rotation angle; increases every frame
    shield_on    = True   # toggled by 's' key

    # Exponential-moving-average FPS counter (α = 0.1)
    fps_ema     = 0.0
    prev_time   = time.perf_counter()

    print("[INFO] Doctor Strange Magic Shield running.")
    print("       Press  s  to toggle | Press  q  to quit.")

    # ---- Main loop ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed – retrying …")
            continue

        # Mirror so the user's right hand appears on their right side
        frame = cv2.flip(frame, 1)

        # ---- Hand detection ------------------------------------------------
        hands = tracker.get_hand_landmarks(frame)

        # ---- Shield rendering ----------------------------------------------
        palm_open_flags = []

        if shield_on:
            for hand in hands:
                is_open = tracker.is_palm_open(hand["landmarks"])
                palm_open_flags.append(is_open)

                if is_open:
                    draw_shield(
                        frame,
                        center=hand["center"],
                        radius=hand["radius"],
                        angle=angle,
                        color=SHIELD_COLOR,
                    )
        else:
            # Still track open/closed state for the HUD even when disabled
            palm_open_flags = [
                tracker.is_palm_open(h["landmarks"]) for h in hands
            ]

        # ---- Advance rotation angle ----------------------------------------
        # Using modulo-360 keeps the float small (no precision drift over time)
        angle = (angle + ANGLE_SPEED) % 360.0

        # ---- FPS counter (exponential moving average) ----------------------
        now      = time.perf_counter()
        dt       = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema     = 0.9 * fps_ema + 0.1 * instant_fps   # smooth display

        # ---- HUD overlay ---------------------------------------------------
        _draw_hud(frame, fps_ema, len(hands), shield_on, palm_open_flags)

        # ---- Display -------------------------------------------------------
        cv2.imshow("Doctor Strange Shield", frame)

        # ---- Key handling --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting …")
            break

        elif key == ord("s"):
            shield_on = not shield_on
            state_str = "enabled" if shield_on else "disabled"
            print(f"[INFO] Shield {state_str}.")

    # ---- Cleanup -----------------------------------------------------------
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
