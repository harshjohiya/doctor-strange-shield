"""
hand_tracking.py
----------------
MediaPipe Hands wrapper for detecting hand landmarks and classifying
whether a palm is open (all four fingers extended).

Returns structured per-hand data used by the shield renderer and
the main application loop.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (used throughout this module)
# ---------------------------------------------------------------------------
WRIST          = 0
THUMB_CMC      = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP      = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP     = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP       = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP      = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# Landmark 9 (middle-finger MCP) is used as the palm centre
PALM_CENTER_IDX = MIDDLE_MCP

# Pairs of (finger-tip, finger-PIP) used for the "finger extended" test
_FINGER_EXT_PAIRS = [
    (INDEX_TIP,  INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP,   RING_PIP),
    (PINKY_TIP,  PINKY_PIP),
]


# ---------------------------------------------------------------------------
# HandTracker  (reusable class; create once and call each frame)
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Thin wrapper around ``mediapipe.solutions.hands`` that converts the raw
    normalised landmarks into pixel-space coordinates and computes per-hand
    metadata consumed by the shield renderer.

    Parameters
    ----------
    max_hands : int
        Maximum number of hands to detect (1 or 2).
    detection_confidence : float
        Minimum confidence for initial hand detection.
    tracking_confidence : float
        Minimum confidence for landmark tracking between frames.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.70,
        tracking_confidence: float = 0.60,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hand_landmarks(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run MediaPipe inference on *frame* and return per-hand data.

        The input frame is expected in **BGR** colour order (straight from
        ``cv2.VideoCapture``).

        Parameters
        ----------
        frame : np.ndarray
            Full-resolution BGR camera frame.

        Returns
        -------
        list of dict, each containing:
            ``landmarks``  – list of 21 ``(x, y)`` pixel-coordinate tuples  
            ``center``     – ``(x, y)`` palm centre (landmark 9)  
            ``radius``     – estimated shield radius derived from hand size
        """
        h, w = frame.shape[:2]

        # MediaPipe requires RGB; we flip colour channels in-place for speed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hands_data: List[Dict[str, Any]] = []

        if not results.multi_hand_landmarks:
            return hands_data

        for hand_lms in results.multi_hand_landmarks:
            # Convert normalised [0,1] coordinates → pixel coordinates
            landmarks = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_lms.landmark
            ]

            center = landmarks[PALM_CENTER_IDX]

            # ---- Shield radius: scale with apparent hand size ----------
            # Distance from wrist to middle-MCP scales linearly with the
            # actual physical distance of the hand from the camera lens.
            wrist_pt  = np.array(landmarks[WRIST],          dtype=np.float32)
            palm_pt   = np.array(landmarks[PALM_CENTER_IDX], dtype=np.float32)
            hand_size = float(np.linalg.norm(palm_pt - wrist_pt))

            # Multiplier chosen so a typical close-up hand gives ~140 px radius
            radius = max(int(hand_size * 2.4), 60)

            hands_data.append({
                "landmarks": landmarks,
                "center":    center,
                "radius":    radius,
            })

        return hands_data

    def is_palm_open(self, landmarks: List[tuple]) -> bool:
        """
        Return ``True`` when the hand is in an open-palm pose (all four
        fingers extended and roughly pointing upward/outward).

        The check is intentionally lenient: at least **4 out of 4** fingers
        must be extended.  The thumb is not tested because its orientation is
        ambiguous on a mirrored webcam feed.

        Parameters
        ----------
        landmarks : list of (x, y) tuples
            Pixel-space landmark list returned by :meth:`get_hand_landmarks`.

        Returns
        -------
        bool
        """
        extended = 0
        for tip_idx, pip_idx in _FINGER_EXT_PAIRS:
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            # "Extended" ⟺ fingertip is above (smaller y) its PIP joint.
            # Works robustly for palms facing the camera and oriented upright.
            if tip_y < pip_y:
                extended += 1

        return extended >= 4

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
