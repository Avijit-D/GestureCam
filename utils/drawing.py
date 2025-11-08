# utils/drawing.py
"""Helper functions for drawing hand landmarks and overlays."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple
import time

import cv2
import numpy as np

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm connections
)

# Small utility
def _rounded_rect(img, top_left, bottom_right, color, radius=12, thickness=-1, alpha=1.0):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return img
    # draw filled rect with rounded corners using circles & rects
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def draw_landmarks(
    frame: np.ndarray,
    hand_landmarks: Iterable[Sequence[Tuple[float, float, float]]],
    mirrored: bool = False,
) -> np.ndarray:
    """Render MediaPipe-style landmark points on the frame.

    Args:
        frame: The frame to draw on
        hand_landmarks: Iterable of landmark sequences with normalized coordinates (x, y, z)
        mirrored: If True, mirror the x coordinates horizontally
    """
    output = frame.copy()
    height, width = output.shape[:2]

    # Draw subtle palm background for readability
    for landmarks in hand_landmarks:
        if mirrored:
            points = [
                (int((1.0 - x) * width), int(y * height))
                for x, y, _ in landmarks
            ]
        else:
            points = [
                (int(x * width), int(y * height))
                for x, y, _ in landmarks
            ]
        if not points:
            continue

        # Palm centroid for subtle translucent fill
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        palm_radius = int(max(40, min(width, height) * 0.06))
        overlay = output.copy()
        cv2.circle(overlay, (cx, cy), palm_radius, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

        # Draw connections (reduced thickness for better performance)
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(output, points[start], points[end], (0, 200, 120), 1, lineType=cv2.LINE_AA)

        # Draw joints with gradient-like color (reduced size for better performance)
        for i, point in enumerate(points):
            color = (int(200 - (i * 4) % 180), 120, 255)  # varied color for better visibility
            cv2.circle(output, point, 3, color, -1, lineType=cv2.LINE_AA)
            # small white center
            cv2.circle(output, point, 1, (255, 255, 255), -1)

    return output

# Cache for text rendering optimization (updates every 3 frames)
_overlay_cache: Dict[Tuple[Tuple[int, int], Tuple[str, ...]], np.ndarray] = {}
_overlay_cache_frame_count = 0
_last_cache_cleanup = time.time()

def overlay_gestures(
    frame: np.ndarray,
    gestures: Sequence[str],
    origin: Tuple[int, int] = (20, 30),
    cache_key: Optional[str] = None,
) -> np.ndarray:
    """
    Draw gesture labels on the frame at a fixed origin.
    Uses caching to reduce redraw overhead - updates every 3 frames.
    Draws a translucent rounded panel for clarity.
    """
    global _overlay_cache, _overlay_cache_frame_count, _last_cache_cleanup

    _overlay_cache_frame_count += 1
    should_update_cache = _overlay_cache_frame_count % 3 == 0

    cache_key_tuple = (origin, tuple(gestures))

    # Periodic cache cleanup
    if time.time() - _last_cache_cleanup > 30:
        # keep cache small
        _overlay_cache = {k: v for i, (k, v) in enumerate(_overlay_cache.items()) if i < 8}
        _last_cache_cleanup = time.time()

    if not should_update_cache and cache_key_tuple in _overlay_cache:
        cached_overlay = _overlay_cache[cache_key_tuple]
        output = frame.copy()
        h, w = cached_overlay.shape[:2]
        x, y = origin
        end_y = min(y + h, output.shape[0])
        end_x = min(x + w, output.shape[1])
        if end_y > y and end_x > x:
            output[y:end_y, x:end_x] = cached_overlay[:end_y - y, :end_x - x]
        return output

    # Build overlay region
    output = frame.copy()
    x, y = origin
    padding_x = 14
    padding_y = 10
    line_h = 28
    total_h = padding_y * 2 + line_h * max(1, len(gestures))
    total_w = max((cv2.getTextSize(g, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for g in gestures), default=120) + padding_x * 2

    # Draw rounded translucent background
    panel_color = (30, 30, 30)
    _rounded_rect(output, (x - 8, y - 8), (x + total_w + 8, y + total_h + 8), panel_color, radius=12, alpha=0.6)

    # Draw each gesture label
    yy = y + padding_y + 6
    for gesture in gestures:
        cv2.putText(output, gesture, (x + padding_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, lineType=cv2.LINE_AA)
        yy += line_h

    # store cache
    try:
        cropped = output[y - 8: y + total_h + 8, x - 8: x + total_w + 8].copy()
        if len(_overlay_cache) < 12:
            _overlay_cache[cache_key_tuple] = cropped
    except Exception:
        pass

    return output

def draw_mode_banner(
    frame: np.ndarray,
    text: str,
    *,
    color: Tuple[int, int, int] = (56, 142, 60),
    alpha: float = 0.85,
) -> np.ndarray:
    """Overlay a semi-transparent banner at the top-left with mode text. Polished visually."""
    output = frame.copy()
    padding = 12
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    width = text_size[0] + padding * 2 + 30
    height = text_size[1] + padding * 2

    # shadow
    shadow = output.copy()
    _rounded_rect(shadow, (10 + 3, 10 + 3), (10 + width + 3, 10 + height + 3), (10, 10, 10), radius=14, alpha=0.35)
    cv2.addWeighted(shadow, 0.6, output, 0.4, 0, output)

    _rounded_rect(output, (10, 10), (10 + width, 10 + height), color, radius=14, alpha=alpha)
    cv2.putText(
        output,
        text,
        (10 + padding, 10 + padding + text_size[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return output

def draw_countdown(frame: np.ndarray, remaining: float) -> np.ndarray:
    """Draw a prominent countdown number at the center of the frame with subtle animation."""
    output = frame.copy()
    height, width = output.shape[:2]
    seconds = max(int(round(remaining)), 0)
    text = str(seconds)
    # animate font slightly with remaining fract
    fract = remaining - int(remaining)
    scale = 1.0 + 0.25 * (1.0 - fract)
    font_scale = min(width, height) / 300 * scale
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 6)
    origin = (
        (width - text_size[0]) // 2,
        (height + text_size[1]) // 2,
    )
    # halo
    for offset in range(6, 2, -2):
        cv2.putText(output, text, (origin[0], origin[1] + offset), cv2.FONT_HERSHEY_DUPLEX, font_scale, (10, 10, 10), 10, lineType=cv2.LINE_AA)
    cv2.putText(
        output,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 180, 255),
        6,
        lineType=cv2.LINE_AA,
    )
    return output

def draw_review_prompts(
    frame: np.ndarray,
    prompts: Sequence[str],
    origin: Tuple[int, int] = (20, 60),
) -> np.ndarray:
    """Display review action prompts on the captured frame with clear layout."""
    output = frame.copy()
    x, y = origin
    # background bar
    total_w = max((cv2.getTextSize(p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] for p in prompts), default=200) + 40
    total_h = 40 * len(prompts)
    _rounded_rect(output, (x - 12, y - 12), (x + total_w + 12, y + total_h + 12), (20, 20, 20), radius=14, alpha=0.6)
    yy = y
    for prompt in prompts:
        cv2.putText(output, prompt, (x, yy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 200), 2, lineType=cv2.LINE_AA)
        yy += 40
    return output

def draw_hold_progress(
    frame: np.ndarray,
    progress: float,
    center: Tuple[int, int] | None = None,
    radius: int = 45,
) -> np.ndarray:
    """Draw a circular progress indicator representing gesture hold duration with smoother arc."""
    progress = float(np.clip(progress, 0.0, 1.0))
    output = frame.copy()
    height, width = output.shape[:2]
    if center is None:
        center = (width - radius - 32, radius + 24)

    base_color = (100, 100, 100)
    progress_color = (10, 200, 220)

    # background ring
    cv2.circle(output, center, radius, base_color, 6, lineType=cv2.LINE_AA)

    if progress > 0.001:
        start_angle = -90
        end_angle = start_angle + int(progress * 360)
        cv2.ellipse(
            output,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            progress_color,
            8,
            lineType=cv2.LINE_AA,
        )

    label = f"{int(progress * 100):d}%"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_origin = (
        center[0] - text_size[0] // 2,
        center[1] + text_size[1] // 2,
    )
    cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, lineType=cv2.LINE_AA)
    return output

def build_composite_frame(
    camera_feed: np.ndarray,
    ui_padding: int = 40,
    background_color: Tuple[int, int, int] = (18, 18, 20),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Build a composite frame centered around the camera feed.
    For best fullscreen feel this returns a composite that matches the camera feed size
    except for a small padding for overlays; the viewport will fill the composite.
    """
    feed_height, feed_width = camera_feed.shape[:2]
    # Keep composite same size as feed (to avoid odd positioned small frame).
    composite_width = feed_width
    composite_height = feed_height

    # Create composite frame with background color same dtype as feed
    if len(camera_feed.shape) == 2:
        bg_value = int(sum(background_color) / 3)
        composite = np.full((composite_height, composite_width), bg_value, dtype=camera_feed.dtype)
    else:
        composite = np.full((composite_height, composite_width, 3), background_color, dtype=camera_feed.dtype)

    # Place camera feed centered (here, it fills the composite)
    viewport_x = 0
    viewport_y = 0
    viewport_width = feed_width
    viewport_height = feed_height
    composite[viewport_y:viewport_y + viewport_height, viewport_x:viewport_x + viewport_width] = camera_feed

    viewport_rect = (viewport_x, viewport_y, viewport_width, viewport_height)
    return composite, viewport_rect
