"""Entry point for gesture-driven camera controller.

Usage:
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import enum
import logging
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from camera_controller import CameraController
from gesture_detector import GestureRecognition, HandGesture
from utils.drawing import (
    build_composite_frame,
    draw_countdown,
    draw_hold_progress,
    draw_landmarks,
    draw_mode_banner,
    draw_review_prompts,
    overlay_gestures,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class Mode(enum.Enum):
    PREVIEW = "Preview"
    GESTURE = "Gesture Mode"
    COUNTDOWN = "Capture Countdown"
    REVIEW = "Review"


MODE_ALLOWED_GESTURES: dict[Mode, Set[str]] = {
    Mode.PREVIEW: {"thumbs_up"},
    Mode.GESTURE: {"two_finger_v", "thumbs_down", "index_only", "pinky_only"},
    Mode.COUNTDOWN: set(),
    Mode.REVIEW: {"thumbs_up", "thumbs_down", "open_palm", "rock_sign"},
}


GESTURE_COOLDOWNS = {
    "open_palm": 0.8,
    "two_finger_v": 1.0,
    "rock_sign": 0.5,
    "index_only": 0.5,
    "pinky_only": 0.5,
    "thumbs_up": 1.0,
    "thumbs_down": 1.0,
}

# OPTIMIZED TUNABLE CONSTANTS
FRAMES_REQUIRED_TO_CONFIRM = 4  # Reduced from 6 for faster response
FRAMES_REQUIRED_TO_RESET = 8  # Increased from 5 to prevent premature resets
STABILITY_WINDOW_FRAMES = 3  # Reduced from 4 for faster response
STABILITY_THRESHOLD_NORM = 0.12  # Increased from 0.06 to be more tolerant
HOLD_DURATION_REQUIRED = 0.3  # Reduced from 0.4 for faster mode switch
PHOTO_SAVED_MESSAGE_DURATION = 2.0
TARGET_FPS = 60  # Increased for better responsiveness
DETECTION_WIDTH = 480  # Smaller detection resolution for faster processing

# Backwards-compatible aliases
frames_required_to_confirm = FRAMES_REQUIRED_TO_CONFIRM
frames_required_to_reset = FRAMES_REQUIRED_TO_RESET
stability_window_frames = STABILITY_WINDOW_FRAMES
stability_threshold_norm = STABILITY_THRESHOLD_NORM
hold_duration_required = HOLD_DURATION_REQUIRED
photo_saved_message_duration = PHOTO_SAVED_MESSAGE_DURATION


def apply_digital_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Crop from the center and resize back to simulate digital zoom."""
    if zoom_factor <= 1.0:
        return frame

    height, width = frame.shape[:2]
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    if new_width <= 0 or new_height <= 0:
        return frame

    start_x = max((width - new_width) // 2, 0)
    start_y = max((height - new_height) // 2, 0)
    end_x = start_x + new_width
    end_y = start_y + new_height

    cropped = frame[start_y:end_y, start_x:end_x]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def prettify_labels(labels: Iterable[str]) -> List[str]:
    return [label.replace("_", " ").title() for label in labels]




def run(camera_index: int = 0) -> None:
    # Use DirectShow backend on Windows for better performance
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    # Set camera resolution and verify it's actually set
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify the actual resolution (camera may not support requested resolution)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera resolution: {actual_width}x{actual_height}")

    controller = CameraController()
    detector = GestureRecognition()

    current_mode = Mode.PREVIEW
    countdown_start: float | None = None
    countdown_duration = 3.0
    last_gesture_trigger: dict[str, float] = {}
    review_frame: np.ndarray | None = None
    status_messages: List[str] = []
    photo_saved_timestamp: Optional[float] = None
    status_message_timestamp: Optional[float] = None
    status_message_duration = 1.5  # seconds
    cooldown_until: float = 0.0
    hold_start_time: Optional[float] = None
    holding_hand_id: Optional[str] = None
    hold_progress: float = 0.0

    # Multi-frame stability tracking
    open_palm_valid_frames: Dict[str, int] = {}
    open_palm_invalid_frames: Dict[str, int] = {}
    wrist_history: Dict[str, deque] = {}

    # Will be updated from actual frame dimensions
    frame_height_px: int = 720
    frame_width_px: int = 1280
    
    # Frame skipping for performance: process detection every 2nd frame, render all
    frame_counter = 0
    gestures: List[HandGesture] = []  # Initialize gestures list for frame skipping
    
    def reset_hold_state() -> None:
        nonlocal hold_start_time, holding_hand_id, hold_progress, status_messages
        hold_start_time = None
        holding_hand_id = None
        hold_progress = 0.0
        status_messages = [m for m in status_messages if m != "Hold steady to switch"]

    def is_hand_stable_multi_frame(
        hand_id: str, wrist_x: float, wrist_y: float, timestamp: float, frame_w: int, frame_h: int
    ) -> bool:
        """Check if hand is stable over the last STABILITY_WINDOW_FRAMES frames."""
        if hand_id not in wrist_history:
            wrist_history[hand_id] = deque(maxlen=stability_window_frames)

        history = wrist_history[hand_id]
        history.append((timestamp, wrist_x, wrist_y))

        if len(history) < 2:
            return True

        total_movement = 0.0
        for i in range(1, len(history)):
            _, x1, y1 = history[i - 1]
            _, x2, y2 = history[i]
            dx_px = abs(x2 - x1) * frame_w
            dy_px = abs(y2 - y1) * frame_h
            movement_px = (dx_px**2 + dy_px**2) ** 0.5
            total_movement += movement_px

        avg_movement_px = total_movement / (len(history) - 1)
        threshold_px = stability_threshold_norm * frame_w
        return avg_movement_px <= threshold_px

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            cooldown_active = timestamp < cooldown_until

            # Get actual frame dimensions (may differ from requested resolution)
            actual_frame_height, actual_frame_width = frame.shape[:2]
            
            # Update frame dimensions from actual captured frame
            frame_height_px = actual_frame_height
            frame_width_px = actual_frame_width

            # Frame skipping: process detection every 2nd frame for better performance
            # But always render all frames for smooth display
            frame_counter += 1
            should_process_detection = (frame_counter % 2 == 0)

            if should_process_detection:
                # OPTIMIZED: Use smaller detection frame (480px width) for faster processing
                # Maintain aspect ratio when downscaling
                detection_scale = DETECTION_WIDTH / frame_width_px
                detection_height = int(frame_height_px * detection_scale)
                detection_frame = cv2.resize(
                    frame, 
                    (DETECTION_WIDTH, detection_height), 
                    interpolation=cv2.INTER_LINEAR
                )

                gestures = detector.classify(detection_frame, timestamp_ms)
            # On skipped frames, reuse previous gestures (they're still valid)
            # This maintains smooth gesture recognition while reducing processing load
            
            gesture_names = sorted({gesture.name for gesture in gestures})
            allowed = MODE_ALLOWED_GESTURES[current_mode]

            def gesture_ready(name: str) -> bool:
                cooldown = GESTURE_COOLDOWNS.get(name, 0.3)
                last_ts = last_gesture_trigger.get(name)
                return last_ts is None or (timestamp - last_ts) >= cooldown

            active_gestures = [g for g in gesture_names if g in allowed and gesture_ready(g)]

            # THUMBS UP DETECTION FOR MODE SWITCHING
            thumbs_up_candidate: Optional[Tuple[float, str]] = None
            if current_mode == Mode.PREVIEW and not cooldown_active:
                current_thumbs_up_hand_ids: Set[str] = set()
                for gesture in gestures:
                    if gesture.name != "thumbs_up" or gesture.hand_id is None:
                        continue

                    hand_id = gesture.hand_id
                    current_thumbs_up_hand_ids.add(hand_id)
                    confidence = getattr(gesture, "confidence", 0.0)

                    # Track thumbs up with simpler validation (thumbs_up is already a stable gesture)
                    if hand_id not in open_palm_valid_frames:
                        open_palm_valid_frames[hand_id] = 0
                    open_palm_valid_frames[hand_id] = open_palm_valid_frames.get(hand_id, 0) + 1
                    open_palm_invalid_frames[hand_id] = 0

                    # If this hand has enough consecutive valid frames, consider as candidate
                    if open_palm_valid_frames[hand_id] >= frames_required_to_confirm:
                        if thumbs_up_candidate is None or confidence > thumbs_up_candidate[0]:
                            thumbs_up_candidate = (confidence, hand_id)

                # Reset counters for hands that didn't appear this frame
                previously_tracked = set(open_palm_valid_frames.keys()) | set(open_palm_invalid_frames.keys())
                missing_hands = previously_tracked - current_thumbs_up_hand_ids
                for hand_id in missing_hands:
                    open_palm_invalid_frames[hand_id] = open_palm_invalid_frames.get(hand_id, 0) + 1
                    if open_palm_invalid_frames[hand_id] >= frames_required_to_reset:
                        open_palm_valid_frames.pop(hand_id, None)
                        open_palm_invalid_frames.pop(hand_id, None)
                        wrist_history.pop(hand_id, None)
                        if holding_hand_id == hand_id:
                            reset_hold_state()

            holding_active = False
            if current_mode == Mode.PREVIEW:
                if cooldown_active:
                    reset_hold_state()
                elif thumbs_up_candidate:
                    _, candidate_id = thumbs_up_candidate
                    if holding_hand_id and holding_hand_id != candidate_id:
                        reset_hold_state()
                    if hold_start_time is None:
                        hold_start_time = timestamp
                        holding_hand_id = candidate_id
                    hold_progress = min((timestamp - hold_start_time) / hold_duration_required, 1.0)
                    holding_active = True
                    if "Hold steady to switch" not in status_messages:
                        status_messages.append("Hold steady to switch")
                    if hold_progress >= 1.0:
                        reset_hold_state()
                        cooldown_until = timestamp + 1.5
                        current_mode = Mode.GESTURE
                        status_messages = ["Gesture Control Mode enabled"]
                        status_message_timestamp = timestamp
                        logger.info("Gesture Control Mode enabled")
                        review_frame = None
                        countdown_start = None
                        active_gestures = []
                        # Keep tracking data for smoother transition
                        continue
                else:
                    # Only reset if no candidate for extended period
                    if hold_start_time is not None:
                        if timestamp - hold_start_time > 0.3:  # 300ms grace period
                            reset_hold_state()
            else:
                reset_hold_state()

            # Ignore gestures during hold, cooldown, or countdown
            if holding_active or cooldown_active or current_mode == Mode.COUNTDOWN:
                active_gestures = []

            log_messages: List[str] = []

            if current_mode == Mode.PREVIEW:
                pass

            elif current_mode == Mode.GESTURE:
                if "two_finger_v" in active_gestures:
                    current_mode = Mode.COUNTDOWN
                    countdown_start = timestamp
                    last_gesture_trigger["two_finger_v"] = timestamp
                    status_messages = ["Countdown started"]
                    status_message_timestamp = timestamp
                    log_messages.append("Photo countdown started")
                elif "thumbs_down" in active_gestures:
                    current_mode = Mode.PREVIEW
                    last_gesture_trigger["thumbs_down"] = timestamp
                    status_messages = ["Preview Mode"]
                    status_message_timestamp = timestamp
                    log_messages.append("Gesture Mode disabled")
                elif "index_only" in active_gestures:
                    last_gesture_trigger["index_only"] = timestamp
                    message = controller.zoom_in()
                    log_messages.append(message)
                    status_messages = [message]
                    status_message_timestamp = timestamp
                elif "pinky_only" in active_gestures:
                    last_gesture_trigger["pinky_only"] = timestamp
                    message = controller.zoom_out()
                    log_messages.append(message)
                    status_messages = [message]
                    status_message_timestamp = timestamp


            elif current_mode == Mode.REVIEW:
                if "thumbs_up" in active_gestures:
                    last_gesture_trigger["thumbs_up"] = timestamp
                    path = controller.save_pending_capture()
                    if path:
                        log_messages.append(f"Saved capture to {path.name}")
                        status_messages = [f"Saved {path.name}"]
                        status_message_timestamp = timestamp
                        photo_saved_timestamp = timestamp
                    else:
                        status_messages = ["No capture to save"]
                        status_message_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "thumbs_down" in active_gestures:
                    last_gesture_trigger["thumbs_down"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Capture discarded"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "open_palm" in active_gestures:
                    last_gesture_trigger["open_palm"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Return to gesture mode"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "rock_sign" in active_gestures:
                    last_gesture_trigger["rock_sign"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Gesture Mode disabled"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.PREVIEW
                    review_frame = None
                    log_messages.append("Gesture Mode disabled from review")

            if current_mode == Mode.COUNTDOWN:
                if countdown_start is None:
                    countdown_start = timestamp
                elapsed = timestamp - countdown_start
                remaining = countdown_duration - elapsed
                if remaining <= 0:
                    countdown_start = None
                    # Mirror frame before capturing
                    mirrored_frame = cv2.flip(frame, 1)
                    zoomed_capture = apply_digital_zoom(mirrored_frame, controller.get_zoom_level())
                    controller.store_capture(zoomed_capture, timestamp)
                    review_frame = controller.peek_pending_capture()
                    current_mode = Mode.REVIEW
                    status_messages = ["Capture ready for review"]
                    status_message_timestamp = timestamp
                    log_messages.append("Capture completed, entering review mode")

            if photo_saved_timestamp is not None:
                elapsed_since_save = timestamp - photo_saved_timestamp
                if elapsed_since_save >= photo_saved_message_duration:
                    status_messages = [msg for msg in status_messages if "Saved" not in msg]
                    photo_saved_timestamp = None

            # Auto-hide status messages after duration
            if status_message_timestamp is not None:
                elapsed_since_status = timestamp - status_message_timestamp
                if elapsed_since_status >= status_message_duration:
                    status_messages = []
                    status_message_timestamp = None

            for message in log_messages:
                logger.info(message)

            # OPTIMIZED: Get landmarks once - use most recent available landmarks
            landmark_entries = detector.get_landmarks()
            # Use landmarks if available (they're already cleared by detector when no hands detected)
            landmarks = [coords.tolist() for _, coords in landmark_entries]

            # === CAMERA FEED LAYER ===
            if current_mode == Mode.REVIEW and review_frame is not None:
                # Review frame is already mirrored when captured
                camera_feed = review_frame.copy()
            else:
                camera_feed = frame.copy()
                # Mirror the camera feed horizontally (selfie view)
                camera_feed = cv2.flip(camera_feed, 1)
                # Draw landmarks with mirrored coordinates to match the mirrored feed
                if landmarks:
                    camera_feed = draw_landmarks(camera_feed, landmarks, mirrored=True)
                camera_feed = apply_digital_zoom(camera_feed, controller.get_zoom_level())

                if current_mode == Mode.COUNTDOWN and countdown_start is not None:
                    elapsed = timestamp - countdown_start
                    remaining = max(countdown_duration - elapsed, 0.0)
                    camera_feed = draw_countdown(camera_feed, remaining)

            # === UI LAYER ===
            composite_frame, viewport_rect = build_composite_frame(camera_feed, ui_padding=100)
            viewport_x, viewport_y, viewport_width, viewport_height = viewport_rect
            composite_height, composite_width = composite_frame.shape[:2]

            left_panel_x = 10
            right_panel_x = viewport_x + viewport_width + 10
            top_panel_y = 10
            bottom_panel_y = viewport_y + viewport_height + 10

            composite_frame = draw_mode_banner(composite_frame, current_mode.value)

            if holding_active:
                composite_frame = draw_hold_progress(
                    composite_frame,
                    hold_progress,
                    center=(composite_width - 65, 65),
                )

            if current_mode not in {Mode.COUNTDOWN, Mode.REVIEW} and not holding_active:
                if gesture_names:
                    composite_frame = overlay_gestures(
                        composite_frame,
                        prettify_labels(gesture_names),
                        origin=(left_panel_x, top_panel_y + 80),
                        cache_key="gesture_names",
                    )
                if status_messages:
                    y_offset = top_panel_y + 80 + (30 * len(gesture_names) if gesture_names else 0)
                    composite_frame = overlay_gestures(
                        composite_frame,
                        status_messages,
                        origin=(left_panel_x, y_offset),
                        cache_key="status_messages",
                    )

            if current_mode == Mode.REVIEW:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "Thumb up: Save",
                        "Thumb down: Discard",
                        "Open palm: Back to Gesture Mode",
                        "Rock sign: Exit Gesture Mode",
                    ],
                    origin=(left_panel_x, top_panel_y + 50),
                )

            cv2.imshow("Gesture Camera Controller", composite_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()