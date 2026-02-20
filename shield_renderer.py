"""
shield_renderer.py
------------------
All drawing and animation logic for the Doctor Strange magic-shield effect.

Design philosophy
~~~~~~~~~~~~~~~~~
* Every draw_* function accepts a pre-allocated ``overlay`` (same shape as
  the camera frame) so callers can accumulate multiple layers before the
  single, final ``cv2.addWeighted`` blend into the real frame.
* Colour is passed in BGR order to match OpenCV conventions.
* Anti-aliased lines (``cv2.LINE_AA``) are used everywhere for smoothness.
* The glow effect is achieved by drawing the same geometry at progressively
  larger radii / thicknesses with decreasing brightness, then doing one
  Gaussian-blur pass before the final blend → smooth halo with no extra
  framebuffer copies per glow ring.
"""

import math
import cv2
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Doctor Strange colour palette  (BGR order)
# ---------------------------------------------------------------------------
GOLD_PRIMARY  = (  0, 165, 255)   # vivid orange-gold
GOLD_BRIGHT   = ( 30, 210, 255)   # bright highlight
GOLD_MID      = (  0, 130, 210)   # mid-tone ring
GOLD_DIM      = (  0,  80, 160)   # background glow
SPARK_WHITE   = (200, 230, 255)   # particle centre
SPARK_ORANGE  = ( 20, 180, 255)   # particle halo


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def draw_glow_circle(
    overlay: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    num_glow_layers: int = 4,
) -> None:
    """
    Draw a glowing circle by rendering concentric rings with decreasing
    brightness outward, then one bright core ring.

    Parameters
    ----------
    overlay       : destination image (drawn in-place)
    center        : (x, y) centre pixel
    radius        : radius of the bright core ring
    color         : BGR colour of the bright core
    thickness     : line thickness of the core ring
    num_glow_layers : number of soft halo rings painted around the core
    """
    # Outer glow rings (darker, thicker → softer)
    for i in range(num_glow_layers, 0, -1):
        scale   = i / num_glow_layers           # 1.0 … 1/N
        dim_bgr = tuple(int(c * scale * 0.45) for c in color)
        r       = radius + i * 4
        t       = thickness + i * 3
        cv2.circle(overlay, center, r, dim_bgr, t, lineType=cv2.LINE_AA)

    # Bright core
    cv2.circle(overlay, center, radius, color, thickness, lineType=cv2.LINE_AA)


def draw_rotating_arcs(
    overlay: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    angle: float,
    color: Tuple[int, int, int],
    num_arcs: int = 8,
    arc_span: int = 28,
    thickness: int = 2,
) -> None:
    """
    Draw evenly-spaced arc segments around a circle that rotate with *angle*.

    Parameters
    ----------
    overlay   : destination image
    center    : (x, y) circle centre
    radius    : circle radius
    angle     : current rotation offset in degrees
    color     : BGR line colour
    num_arcs  : number of arc segments
    arc_span  : angular width of each arc in degrees
    thickness : line thickness
    """
    step = 360 // num_arcs
    for i in range(num_arcs):
        start = int(angle + i * step) % 360
        end   = start + arc_span
        cv2.ellipse(
            overlay, center, (radius, radius),
            0, start, end,
            color, thickness, lineType=cv2.LINE_AA,
        )


def draw_spokes(
    overlay: np.ndarray,
    center: Tuple[int, int],
    inner_r: int,
    outer_r: int,
    angle: float,
    color: Tuple[int, int, int],
    num_spokes: int = 16,
    thickness: int = 1,
) -> None:
    """
    Radial line segments (spokes) between *inner_r* and *outer_r*, rotating
    with *angle*.

    Parameters
    ----------
    overlay   : destination image
    center    : (x, y) centre pixel
    inner_r   : spoke start radius
    outer_r   : spoke end radius
    angle     : rotation offset in degrees
    color     : BGR colour
    num_spokes: number of spokes
    thickness : line thickness
    """
    step = 360.0 / num_spokes
    cx, cy = center
    for i in range(num_spokes):
        theta = math.radians(angle + i * step)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x1 = int(cx + inner_r * cos_t);  y1 = int(cy + inner_r * sin_t)
        x2 = int(cx + outer_r * cos_t);  y2 = int(cy + outer_r * sin_t)
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness,
                 lineType=cv2.LINE_AA)


def draw_particles(
    overlay: np.ndarray,
    center: Tuple[int, int],
    orbit_r: int,
    angle: float,
    outer_color: Tuple[int, int, int],
    inner_color: Tuple[int, int, int],
    num_particles: int = 6,
    particle_r: int = 4,
) -> None:
    """
    Orbiting particles (small filled discs with a bright centre dot) that
    travel along a circular orbit.

    Parameters
    ----------
    overlay       : destination image
    center        : (x, y) orbit centre
    orbit_r       : orbit radius
    angle         : current angle offset in degrees (drives the orbit)
    outer_color   : halo / outer glow colour (BGR)
    inner_color   : bright centre colour (BGR)
    num_particles : number of evenly-spaced particles
    particle_r    : radius of the halo disc
    """
    step = 360.0 / num_particles
    cx, cy = center
    for i in range(num_particles):
        theta = math.radians(angle + i * step)
        px = int(cx + orbit_r * math.cos(theta))
        py = int(cy + orbit_r * math.sin(theta))
        # Outer halo
        cv2.circle(overlay, (px, py), particle_r,     outer_color, -1,
                   lineType=cv2.LINE_AA)
        # Bright core
        cv2.circle(overlay, (px, py), max(particle_r - 2, 1), inner_color, -1,
                   lineType=cv2.LINE_AA)


def draw_geometric_pattern(
    overlay: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    angle: float,
    color: Tuple[int, int, int],
) -> None:
    """
    Inner rotating hexagon + tristar (Star-of-David style lines) that gives
    the shield its mystical, rune-like inner fill.

    Parameters
    ----------
    overlay : destination image
    center  : (x, y) centre
    radius  : outer shield radius (inner pattern is scaled relative to this)
    angle   : rotation angle in degrees
    color   : BGR colour
    """
    cx, cy = center

    # ---- Outer hexagon at 46% of shield radius --------------------------
    hex_r = int(radius * 0.46)
    hex_pts = []
    for i in range(6):
        theta = math.radians(angle + i * 60.0)
        hex_pts.append([
            int(cx + hex_r * math.cos(theta)),
            int(cy + hex_r * math.sin(theta)),
        ])
    hex_pts_np = np.array(hex_pts, np.int32)
    cv2.polylines(overlay, [hex_pts_np], True, color, 1, lineType=cv2.LINE_AA)

    # Tristar: connect vertex i to vertex i+3 (opposite corners)
    for i in range(3):
        cv2.line(overlay, tuple(hex_pts[i]), tuple(hex_pts[i + 3]),
                 color, 1, lineType=cv2.LINE_AA)

    # ---- Inner triangle at 24% of shield radius (counter-rotates) -------
    tri_r = int(radius * 0.24)
    tri_pts = []
    for i in range(3):
        theta = math.radians(-angle * 1.5 + i * 120.0)
        tri_pts.append([
            int(cx + tri_r * math.cos(theta)),
            int(cy + tri_r * math.sin(theta)),
        ])
    tri_pts_np = np.array(tri_pts, np.int32)
    cv2.polylines(overlay, [tri_pts_np], True, GOLD_BRIGHT, 1,
                  lineType=cv2.LINE_AA)

    # Small centre dot
    cv2.circle(overlay, center, 3, SPARK_WHITE, -1, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def draw_shield(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    angle: float,
    color: Tuple[int, int, int] = GOLD_PRIMARY,
) -> None:
    """
    Composite the full Doctor Strange magic-shield effect onto *frame*
    (modified **in-place**).

    Shield anatomy (inside-out):
    
    1. Bright centre dot
    2. Inner triangle (counter-rotates)
    3. Inner hexagon + tristar geometric pattern
    4. Inner glow ring
    5. Radial spokes (slow clockwise rotation)
    6. Inner orbiting particles (fast counter-clockwise)
    7. Mid-ring with counter-rotating arc segments
    8. Outer glow ring with clockwise arc segments
    9. Outer orbiting particles (medium clockwise)
   10. Gaussian-blur glow pass blended additively

    Parameters
    ----------
    frame  : BGR camera frame to draw on
    center : (x, y) shield centre (palm landmark 9 in pixel coordinates)
    radius : shield outer radius (derived from hand size in the tracker)
    angle  : ever-increasing rotation angle in degrees (drives all motion)
    color  : primary BGR colour  (default: orange-gold)
    """
    cx, cy = center
    h, w   = frame.shape[:2]

    # ----- Bounds guard: skip hands that are partially off-screen ----------
    margin = radius + 20
    if cx < -margin or cx > w + margin or cy < -margin or cy > h + margin:
        return

    # ----- Allocate a black overlay; all shield geometry goes here ---------
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # -----------------------------------------------------------------------
    # Layer 1 – Outer glow ring (clockwise arc segments + full halo)
    # -----------------------------------------------------------------------
    draw_glow_circle(
        overlay, center, radius,
        color=color, thickness=2, num_glow_layers=4,
    )
    draw_rotating_arcs(
        overlay, center, radius,
        angle=angle, color=GOLD_BRIGHT,
        num_arcs=8, arc_span=26, thickness=2,
    )

    # -----------------------------------------------------------------------
    # Layer 2 – Mid ring (counter-rotates; slightly smaller)
    # -----------------------------------------------------------------------
    mid_r = int(radius * 0.72)
    draw_glow_circle(
        overlay, center, mid_r,
        color=GOLD_MID, thickness=1, num_glow_layers=2,
    )
    draw_rotating_arcs(
        overlay, center, mid_r,
        angle=-angle * 1.4, color=GOLD_PRIMARY,
        num_arcs=6, arc_span=32, thickness=2,
    )

    # -----------------------------------------------------------------------
    # Layer 3 – Inner ring
    # -----------------------------------------------------------------------
    inner_r = int(radius * 0.46)
    draw_glow_circle(
        overlay, center, inner_r,
        color=GOLD_BRIGHT, thickness=1, num_glow_layers=2,
    )

    # -----------------------------------------------------------------------
    # Layer 4 – Radial spokes (slow clockwise)
    # -----------------------------------------------------------------------
    draw_spokes(
        overlay, center,
        inner_r=inner_r, outer_r=radius,
        angle=angle * 0.4, color=GOLD_DIM,
        num_spokes=16, thickness=1,
    )

    # -----------------------------------------------------------------------
    # Layer 5 – Geometric inner pattern (hex + tristar + mini-triangle)
    # -----------------------------------------------------------------------
    draw_geometric_pattern(overlay, center, radius, angle, GOLD_BRIGHT)

    # -----------------------------------------------------------------------
    # Layer 6 – Outer orbiting particles (medium speed, clockwise)
    # -----------------------------------------------------------------------
    draw_particles(
        overlay, center,
        orbit_r=int(radius * 1.04),
        angle=angle * 1.8,
        outer_color=SPARK_ORANGE,
        inner_color=SPARK_WHITE,
        num_particles=8,
        particle_r=4,
    )

    # -----------------------------------------------------------------------
    # Layer 7 – Inner orbiting particles (fast, counter-clockwise)
    # -----------------------------------------------------------------------
    draw_particles(
        overlay, center,
        orbit_r=int(radius * 0.59),
        angle=-angle * 2.8,
        outer_color=GOLD_BRIGHT,
        inner_color=SPARK_WHITE,
        num_particles=5,
        particle_r=3,
    )

    # -----------------------------------------------------------------------
    # Final compositing
    # -----------------------------------------------------------------------
    # Pass 1 – Crisp additive blend: brights punch through the background
    cv2.addWeighted(frame, 1.0, overlay, 0.90, 0, dst=frame)

    # Pass 2 – Blurred glow halo (soft bloom around every drawn element)
    glow = cv2.GaussianBlur(overlay, (11, 11), 0)
    cv2.addWeighted(frame, 1.0, glow, 0.40, 0, dst=frame)
