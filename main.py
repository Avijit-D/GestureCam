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
    draw_countdown,
    draw_hold_progress,
    draw_landmarks,
    draw_mode_banner,
    draw_review_prompts,
    overlay_gestures,
    draw_edit_ui,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class Mode(enum.Enum):
    PREVIEW = "Preview"
    GESTURE = "Gesture Mode"
    COUNTDOWN = "Capture Countdown"
    REVIEW = "Review"
    EDIT = "Edit Mode"


MODE_ALLOWED_GESTURES: dict[Mode, Set[str]] = {
    Mode.PREVIEW: {"thumbs_up"},
    Mode.GESTURE: {"two_finger_v", "thumbs_down", "index_only", "pinky_only"},
    Mode.COUNTDOWN: set(),
    Mode.REVIEW: {"thumbs_up", "thumbs_down", "open_palm", "rock_sign"},
    Mode.EDIT: {"open_palm", "fist", "index_only"},  # thumbs_up/thumbs_down disabled in edit mode
}


GESTURE_COOLDOWNS = {
    "open_palm": 0.8,
    "two_finger_v": 1.0,
    "rock_sign": 1.5,
    "index_only": 0.5,
    "pinky_only": 0.5,
    "thumbs_up": 1.0,
    "thumbs_down": 1.0,
    "fist": 0.3,
}

# OPTIMIZED TUNABLE CONSTANTS
FRAMES_REQUIRED_TO_CONFIRM = 4
FRAMES_REQUIRED_TO_RESET = 8
STABILITY_WINDOW_FRAMES = 3
STABILITY_THRESHOLD_NORM = 0.12
HOLD_DURATION_REQUIRED = 0.3
PHOTO_SAVED_MESSAGE_DURATION = 2.0
TARGET_FPS = 60
DETECTION_WIDTH = 480

# Display window size (fixed regardless of camera resolution)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

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


def apply_all_edits(image: np.ndarray, params: Dict[str, int]) -> np.ndarray:
    """Apply all edit parameters to an image."""
    result = image.copy().astype(np.float32)
    
    # Apply brightness
    if params["brightness"] != 0:
        result = result + params["brightness"]
    
    # Apply contrast (alpha scaling)
    if params["contrast"] != 0:
        alpha = 1.0 + (params["contrast"] / 100.0)  # -50 to +50 -> 0.5 to 1.5
        result = result * alpha
    
    # Clip after brightness and contrast
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Apply saturation
    if params["saturation"] != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + params["saturation"], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Apply warmth (shift red/blue channels)
    if params["warmth"] != 0:
        warmth_factor = params["warmth"] * 0.6  # Scale down for subtlety
        result = result.astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] + warmth_factor, 0, 255)  # Red
        result[:, :, 0] = np.clip(result[:, :, 0] - warmth_factor, 0, 255)  # Blue
        result = result.astype(np.uint8)
    
    # Apply sharpness
    if params["sharpness"] != 0:
        if params["sharpness"] > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            multiplier = params["sharpness"] / 50.0  # 0 to 1
            sharpened = cv2.filter2D(result, -1, kernel * multiplier)
            result = cv2.addWeighted(result, 1.0 - multiplier, sharpened, multiplier, 0)
        else:
            # Negative sharpness = blur
            blur_amount = abs(params["sharpness"]) // 10 * 2 + 1  # Odd numbers 1, 3, 5...
            result = cv2.GaussianBlur(result, (blur_amount, blur_amount), 0)
    
    return result


def run(camera_index: int = 0) -> None:
    # Use DirectShow backend on Windows for better performance
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    # Set camera resolution and verify it's actually set
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify the actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera native resolution: {actual_width}x{actual_height}")
    logger.info(f"Display window size: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    # Create window with fixed size BEFORE loop
    cv2.namedWindow("Gesture Camera Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Camera Controller", DISPLAY_WIDTH, DISPLAY_HEIGHT)

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
    status_message_duration = 1.5
    cooldown_until: float = 0.0
    hold_start_time: Optional[float] = None
    holding_hand_id: Optional[str] = None
    hold_progress: float = 0.0

    # Edit mode state
    edit_params: Dict[str, int] = {"brightness": 0, "contrast": 0, "saturation": 0, "warmth": 0, "sharpness": 0}
    selected_param: Optional[str] = None
    param_hover_start: Optional[float] = None
    param_hover_target: Optional[str] = None
    original_edit_frame: Optional[np.ndarray] = None
    
    # Two-hand gesture validation (to prevent false positives)
    two_hand_gesture_frames: Dict[str, int] = {}  # Track consecutive frames for each gesture

    # Multi-frame stability tracking
    open_palm_valid_frames: Dict[str, int] = {}
    open_palm_invalid_frames: Dict[str, int] = {}
    wrist_history: Dict[str, deque] = {}

    # Frame dimensions (use fixed display size)
    frame_height_px: int = DISPLAY_HEIGHT
    frame_width_px: int = DISPLAY_WIDTH
    
    # Frame skipping for performance
    frame_counter = 0
    gestures: List[HandGesture] = []
    
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

            # Resize to fixed display size IMMEDIATELY
            if frame.shape[0] != DISPLAY_HEIGHT or frame.shape[1] != DISPLAY_WIDTH:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), 
                                 interpolation=cv2.INTER_LINEAR)

            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            cooldown_active = timestamp < cooldown_until

            # Frame skipping: process detection every 2nd frame
            frame_counter += 1
            should_process_detection = (frame_counter % 2 == 0)

            if should_process_detection:
                detection_scale = DETECTION_WIDTH / frame_width_px
                detection_height = int(frame_height_px * detection_scale)
                detection_frame = cv2.resize(
                    frame, 
                    (DETECTION_WIDTH, detection_height), 
                    interpolation=cv2.INTER_LINEAR
                )

                gestures = detector.classify(detection_frame, timestamp_ms)
            
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

                    if hand_id not in open_palm_valid_frames:
                        open_palm_valid_frames[hand_id] = 0
                    open_palm_valid_frames[hand_id] = open_palm_valid_frames.get(hand_id, 0) + 1
                    open_palm_invalid_frames[hand_id] = 0

                    if open_palm_valid_frames[hand_id] >= frames_required_to_confirm:
                        if thumbs_up_candidate is None or confidence > thumbs_up_candidate[0]:
                            thumbs_up_candidate = (confidence, hand_id)

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
                        continue
                else:
                    if hold_start_time is not None:
                        if timestamp - hold_start_time > 0.3:
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
                if "open_palm" in active_gestures:
                    last_gesture_trigger["open_palm"] = timestamp
                    log_messages.append("✋ Open palm → Entering Edit Mode")
                    # Enter edit mode
                    current_mode = Mode.EDIT
                    status_messages = ["Edit Mode"]
                    status_message_timestamp = timestamp
                    # Initialize edit state
                    edit_params = {"brightness": 0, "contrast": 0, "saturation": 0, "warmth": 0, "sharpness": 0}
                    selected_param = None
                    param_hover_start = None
                    param_hover_target = None
                elif "thumbs_up" in active_gestures:
                    last_gesture_trigger["thumbs_up"] = timestamp
                    log_messages.append("👍 Thumbs up → Saving photo")
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
                    log_messages.append("👎 Thumbs down → Discarding photo")
                    controller.discard_pending_capture()
                    status_messages = ["Capture discarded"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "rock_sign" in active_gestures:
                    last_gesture_trigger["rock_sign"] = timestamp
                    log_messages.append("🤘 Rock sign → Exiting Gesture Mode")
                    controller.discard_pending_capture()
                    status_messages = ["Gesture Mode disabled"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.PREVIEW
                    review_frame = None

            elif current_mode == Mode.EDIT:
                # Store original frame if not already stored
                if original_edit_frame is None and review_frame is not None:
                    original_edit_frame = review_frame.copy()
                
                # Get landmarks for gesture control
                landmark_entries = detector.get_landmarks()
                landmarks = [coords.tolist() for _, coords in landmark_entries]
                
                # Get wrist position for parameter selection (more reliable than index finger)
                wrist_pos = None
                if landmarks:
                    for hand_landmarks in landmarks:
                        if len(hand_landmarks) > 0:
                            wrist_pos = hand_landmarks[0][:2]  # Wrist position (normalized)
                            break
                
                # Detect fist gesture for slider control
                is_fist = "fist" in gesture_names
                
                # Check for two-hand gestures (EDIT mode only)
                two_hand_gesture = detector.detect_two_hand_gesture(gestures)
                
                # Multi-frame validation for two-hand gestures (require 3 consecutive frames)
                REQUIRED_TWO_HAND_FRAMES = 3
                
                if two_hand_gesture:
                    # Increment frame count for this gesture
                    two_hand_gesture_frames[two_hand_gesture] = two_hand_gesture_frames.get(two_hand_gesture, 0) + 1
                    # Reset other gesture counts
                    for other_gesture in ["two_open_palm", "two_fist"]:
                        if other_gesture != two_hand_gesture:
                            two_hand_gesture_frames[other_gesture] = 0
                else:
                    # Reset all counts if no two-hand gesture detected
                    two_hand_gesture_frames.clear()
                
                # Only trigger if gesture detected for required frames
                if two_hand_gesture == "two_open_palm" and two_hand_gesture_frames.get("two_open_palm", 0) >= REQUIRED_TWO_HAND_FRAMES:
                    # SAVE edited photo (two open palms = save in edit mode)
                    last_gesture_trigger["two_open_palm"] = timestamp
                    log_messages.append("✋✋ Two open palms → Saving edited photo")
                    two_hand_gesture_frames.clear()  # Reset after triggering
                    if original_edit_frame is not None:
                        edited_frame = apply_all_edits(original_edit_frame, edit_params)
                        controller.store_capture(edited_frame, timestamp)
                        path = controller.save_pending_capture()
                        if path:
                            log_messages.append(f"Saved edited capture to {path.name}")
                            status_messages = [f"Saved {path.name}"]
                            status_message_timestamp = timestamp
                            photo_saved_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                    original_edit_frame = None
                    selected_param = None
                    edit_params = {"brightness": 0, "contrast": 0, "saturation": 0, "warmth": 0, "sharpness": 0}
                    
                elif two_hand_gesture == "two_fist" and two_hand_gesture_frames.get("two_fist", 0) >= REQUIRED_TWO_HAND_FRAMES:
                    # DISCARD edits (two fists = discard in edit mode)
                    last_gesture_trigger["two_fist"] = timestamp
                    log_messages.append("✊✊ Two fists → Discarding edits")
                    two_hand_gesture_frames.clear()  # Reset after triggering
                    controller.discard_pending_capture()
                    status_messages = ["Edits discarded"]
                    status_message_timestamp = timestamp
                    current_mode = Mode.GESTURE
                    review_frame = None
                    original_edit_frame = None
                    selected_param = None
                    edit_params = {"brightness": 0, "contrast": 0, "saturation": 0, "warmth": 0, "sharpness": 0}
                
                # ===== PARAMETER SELECTION VIA WRIST POSITION =====
                param_names = ["brightness", "contrast", "saturation", "warmth", "sharpness"]
                
                # Box layout (normalized coordinates)
                num_boxes = 5
                total_spacing = 0.06
                usable_width = 0.98 - total_spacing
                box_width = usable_width / num_boxes
                box_spacing = total_spacing / (num_boxes - 1)
                box_height = 0.12
                box_y_start = 0.88 - box_height
                
                hovered_param = None
                
                # Use wrist position for parameter selection (more reliable than index finger)
                if wrist_pos:
                    x, y = wrist_pos  # Already normalized (0.0-1.0)
                    
                    # Mirror x coordinate to match the mirrored display
                    x = 1.0 - x
                    
                    # Check if wrist is in bottom region
                    if y > box_y_start:
                        for i, param in enumerate(param_names):
                            box_x_start = 0.01 + i * (box_width + box_spacing)
                            box_x_end = box_x_start + box_width
                            
                            if box_x_start <= x <= box_x_end:
                                hovered_param = param
                                break
                    
                    # Track hover duration for confirmation (2 seconds)
                    if hovered_param:
                        if param_hover_target != hovered_param:
                            param_hover_target = hovered_param
                            param_hover_start = timestamp
                        elif param_hover_start and (timestamp - param_hover_start) >= 2.0:
                            # Toggle selection: if already selected, deselect; otherwise select
                            if selected_param == hovered_param:
                                selected_param = None
                                status_messages = [f"Deselected: {hovered_param.title()}"]
                                status_message_timestamp = timestamp
                                log_messages.append(f"Deselected parameter: {hovered_param}")
                            else:
                                selected_param = hovered_param
                                status_messages = [f"Selected: {selected_param.title()}"]
                                status_message_timestamp = timestamp
                                log_messages.append(f"Selected parameter: {selected_param}")
                            param_hover_start = None
                            param_hover_target = None
                    else:
                        param_hover_start = None
                        param_hover_target = None
                else:
                    param_hover_start = None
                    param_hover_target = None
                
                # Slider control with fist gesture
                if is_fist and selected_param and wrist_pos:
                    wrist_x = wrist_pos[0]
                    
                    # Mirror x coordinate to match the mirrored display
                    wrist_x = 1.0 - wrist_x
                    
                    # Map wrist position to -50 to +50 with dead zone
                    if wrist_x < 0.33:
                        value = int(-50 + (wrist_x / 0.33) * 40)
                    elif wrist_x > 0.67:
                        value = int(10 + ((wrist_x - 0.67) / 0.33) * 40)
                    else:
                        center_offset = (wrist_x - 0.33) / 0.34
                        value = int(-10 + center_offset * 20)
                    
                    value = max(-50, min(50, value))
                    edit_params[selected_param] = value
                
                # Note: thumbs_up and thumbs_down are disabled in EDIT mode
                # Use two-hand gestures instead: two_open_palm = save, two_fist = discard

            if current_mode == Mode.COUNTDOWN:
                if countdown_start is None:
                    countdown_start = timestamp
                elapsed = timestamp - countdown_start
                remaining = countdown_duration - elapsed
                if remaining <= 0:
                    countdown_start = None
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

            if status_message_timestamp is not None:
                elapsed_since_status = timestamp - status_message_timestamp
                if elapsed_since_status >= status_message_duration:
                    status_messages = []
                    status_message_timestamp = None

            for message in log_messages:
                logger.info(message)

            # Get landmarks for display
            landmark_entries = detector.get_landmarks()
            landmarks = [coords.tolist() for _, coords in landmark_entries]

            # === CAMERA FEED LAYER ===
            if current_mode == Mode.REVIEW and review_frame is not None:
                camera_feed = review_frame.copy()
            elif current_mode == Mode.EDIT and original_edit_frame is not None:
                # Apply edits in real-time for preview
                camera_feed = apply_all_edits(original_edit_frame, edit_params)
                
                # Draw ghost hands over the edited photo
                if landmarks:
                    camera_feed = draw_landmarks(camera_feed, landmarks, mirrored=True)
                
                # Draw edit UI with hover progress (2-second countdown)
                hover_progress = 0.0
                if param_hover_start and param_hover_target:
                    hover_progress = min((timestamp - param_hover_start) / 2.0, 1.0)
                camera_feed = draw_edit_ui(camera_feed, edit_params, selected_param, 
                                            param_hover_target, hover_progress)
            else:
                camera_feed = frame.copy()
                camera_feed = cv2.flip(camera_feed, 1)
                if landmarks:
                    camera_feed = draw_landmarks(camera_feed, landmarks, mirrored=True)
                camera_feed = apply_digital_zoom(camera_feed, controller.get_zoom_level())

                if current_mode == Mode.COUNTDOWN and countdown_start is not None:
                    elapsed = timestamp - countdown_start
                    remaining = max(countdown_duration - elapsed, 0.0)
                    camera_feed = draw_countdown(camera_feed, remaining)

            # === UI LAYER ===
            composite_frame = camera_feed.copy()
            composite_height, composite_width = composite_frame.shape[:2]
            viewport_x, viewport_y, viewport_width, viewport_height = (0, 0, composite_width, composite_height)

            left_panel_x = 10
            right_panel_x = viewport_x + viewport_width + 10
            top_panel_y = 10
            bottom_panel_y = viewport_y + viewport_height + 10

            composite_frame = draw_mode_banner(composite_frame, current_mode.value)
            
            # Calculate mode banner height to position gesture menu below it
            # Mode banner is at y=12, with padding=16, text height ~20-25, so total height ~50-60
            mode_banner_height = 60
            gesture_menu_y = 12 + mode_banner_height + 15  # 15px spacing below mode banner

            if holding_active:
                composite_frame = draw_hold_progress(
                    composite_frame,
                    hold_progress,
                    center=(composite_width - 65, 65),
                )

            # Track right side y position for stacking messages
            right_side_y = top_panel_y + 10

            if current_mode not in {Mode.COUNTDOWN, Mode.REVIEW, Mode.EDIT} and not holding_active:
                if gesture_names:
                    # Position current gesture text at top right
                    # Calculate width needed for gesture text box to align it properly from right edge
                    prettified_gestures = prettify_labels(gesture_names)
                    padding_x = 18
                    shadow_offset = 6
                    max_text_width = max((cv2.getTextSize(g, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for g in prettified_gestures), default=120)
                    box_width = max_text_width + padding_x * 2 + shadow_offset * 2
                    right_panel_x = composite_width - box_width - 10  # 10px margin from right edge
                    composite_frame = overlay_gestures(
                        composite_frame,
                        prettified_gestures,
                        origin=(right_panel_x, right_side_y),
                        cache_key="gesture_names",
                    )
                    # Update right side y position for next element
                    # overlay_gestures uses: padding_y*2 + line_h*len + shadow_offset*2
                    padding_y = 12
                    line_h = 32
                    shadow_offset = 6
                    box_height = padding_y * 2 + line_h * len(prettified_gestures) + shadow_offset * 2
                    right_side_y += box_height + 10  # height + spacing
                
                if status_messages:
                    # Position status messages on right side, below gesture names
                    prettified_messages = status_messages
                    padding_x = 18
                    shadow_offset = 6
                    max_text_width = max((cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for m in prettified_messages), default=120)
                    box_width = max_text_width + padding_x * 2 + shadow_offset * 2
                    right_panel_x = composite_width - box_width - 10  # 10px margin from right edge
                    composite_frame = overlay_gestures(
                        composite_frame,
                        prettified_messages,
                        origin=(right_panel_x, right_side_y),
                        cache_key="status_messages",
                    )
                    # Update right side y position for potential future elements
                    padding_y = 12
                    line_h = 32
                    box_height = padding_y * 2 + line_h * len(prettified_messages) + shadow_offset * 2
                    right_side_y += box_height + 10
            else:
                # For COUNTDOWN, REVIEW, EDIT modes, show status messages on right side
                if status_messages:
                    prettified_messages = status_messages
                    padding_x = 18
                    shadow_offset = 6
                    max_text_width = max((cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for m in prettified_messages), default=120)
                    box_width = max_text_width + padding_x * 2 + shadow_offset * 2
                    right_panel_x = composite_width - box_width - 10
                    composite_frame = overlay_gestures(
                        composite_frame,
                        prettified_messages,
                        origin=(right_panel_x, right_side_y),
                        cache_key="status_messages",
                    )

            # Show gesture hints for all modes (on left side, below mode banner, smaller size)
            if current_mode == Mode.PREVIEW:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "Thumbs up: Enter Gesture Mode",
                    ],
                    origin=(left_panel_x, gesture_menu_y),
                    font_scale=0.65,
                    line_height=28,
                )
            elif current_mode == Mode.GESTURE:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "V sign: Take Photo",
                        "Thumbs down: Exit Gesture Mode",
                        "Index finger: Zoom In",
                        "Pinky finger: Zoom Out",
                    ],
                    origin=(left_panel_x, gesture_menu_y),
                    font_scale=0.65,
                    line_height=28,
                )
            elif current_mode == Mode.COUNTDOWN:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "Countdown in progress...",
                    ],
                    origin=(left_panel_x, gesture_menu_y),
                    font_scale=0.65,
                    line_height=28,
                )
            elif current_mode == Mode.REVIEW:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "Thumb up: Save",
                        "Thumb down: Discard",
                        "Open palm: Edit Photo",
                        "Rock sign: Exit Gesture Mode",
                    ],
                    origin=(left_panel_x, gesture_menu_y),
                    font_scale=0.65,
                    line_height=28,
                )
            elif current_mode == Mode.EDIT:
                composite_frame = draw_review_prompts(
                    composite_frame,
                    [
                        "Wrist: Select Parameter",
                        "Fist: Adjust Value",
                        "Two open palms: Save",
                        "Two fists: Discard",
                    ],
                    origin=(left_panel_x, gesture_menu_y),
                    font_scale=0.65,
                    line_height=28,
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