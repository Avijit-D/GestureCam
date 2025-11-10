"""Gesture recognition powered by MediaPipe Hand Landmarker."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV (opencv-python) is required for gesture detection") from exc

try:
    from mediapipe import Image as MPImage
    from mediapipe import ImageFormat
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError("MediaPipe is required for gesture detection. Install mediapipe.") from exc


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass
class HandGesture:
    """Represents a classified gesture with optional metadata."""

    name: str
    confidence: float
    handedness: Optional[str] = None
    hand_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GestureRecognition:
    """Detect hands in frames and classify high-level gestures."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = self._ensure_model_exists(model_path)
        self._landmarker = self._create_landmarker()
        self._result_queue: Deque[Tuple[int, vision.HandLandmarkerResult]] = deque(maxlen=2)  # Reduced for lower latency
        self._hand_history: Dict[str, Deque[Tuple[int, float]]] = defaultdict(lambda: deque(maxlen=10))
        self._last_swipe_timestamp: int = 0
        self._last_landmarks: List[np.ndarray] = []
        self._last_landmark_ids: List[str] = []

    def _ensure_model_exists(self, model_path: Path | str | None) -> Path:
        """Download the MediaPipe model locally if it is absent."""
        default_model = Path("models/hand_landmarker.task")
        path = Path(model_path) if model_path else default_model
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(MODEL_URL, path)
        return path

    def _create_landmarker(self) -> vision.HandLandmarker:
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.4,  # Reduced from 0.5 for better detection
            min_hand_presence_confidence=0.4,   # Reduced from 0.5
            min_tracking_confidence=0.4,         # Reduced from 0.5
            result_callback=self._result_callback,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _result_callback(
        self,
        result: vision.HandLandmarkerResult,
        output_image: MPImage,
        timestamp_ms: int,
    ) -> None:
        self._result_queue.append((timestamp_ms, result))

    def classify(self, frame: np.ndarray, timestamp_ms: int) -> List[HandGesture]:
        """Analyze a frame and return the gestures detected."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        self._landmarker.detect_async(mp_image, timestamp_ms)

        gestures: List[HandGesture] = []
        # Always use the most recent result available, discard older ones
        # This ensures we always render the LATEST hand position, not old queued results
        if self._result_queue:
            # Skip to the last (most recent) result
            while len(self._result_queue) > 1:
                self._result_queue.popleft()
            ts, result = self._result_queue.popleft()
            gestures = self._interpret_result(result, ts)
        return gestures

    def _interpret_result(
        self, result: vision.HandLandmarkerResult, timestamp_ms: int
    ) -> List[HandGesture]:
        gestures: List[HandGesture] = []
        landmark_store: List[np.ndarray] = []
        landmark_ids: List[str] = []
        
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = None
            score = 1.0
            if result.handedness and index < len(result.handedness):
                handedness_category = result.handedness[index][0]
                handedness = handedness_category.category_name.lower()
                score = handedness_category.score

            key = f"{handedness or 'unknown'}_{index}"
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            landmark_store.append(coords)
            landmark_ids.append(key)

            gestures.extend(
                self._classify_static_gestures(coords, handedness, key, score)
            )
            
            swipe = self._detect_swipe(key, coords, timestamp_ms)
            if swipe:
                gestures.append(HandGesture(swipe, 0.9, handedness, hand_id=key))

        # Always update landmarks, even if empty (clears stale data immediately)
        self._last_landmarks = landmark_store
        self._last_landmark_ids = landmark_ids
        return gestures

    def _classify_static_gestures(
        self,
        coords: np.ndarray,
        handedness: Optional[str],
        hand_id: str,
        confidence: float,
    ) -> List[HandGesture]:
        gestures: List[HandGesture] = []

        fingers_extended = {
            finger: self._is_finger_extended(coords, joints, handedness)
            for finger, joints in FINGER_JOINTS.items()
        }
        extended_count = sum(1 for value in fingers_extended.values() if value)

        wrist = coords[0]
        thumb_tip = coords[FINGER_JOINTS["thumb"][3]]
        index_tip = coords[FINGER_JOINTS["index"][3]]
        index_mcp = coords[FINGER_JOINTS["index"][0]]

        non_thumb_extended = sum(
            1 for f in ("index", "middle", "ring", "pinky") if fingers_extended[f]
        )

        pinch_distance = float(np.linalg.norm(thumb_tip[:2] - index_tip[:2]))
        wrist_radius = float(np.linalg.norm(index_mcp[:2] - wrist[:2]))

        # OPTIMIZED: Check open palm first (most common gesture)
        if self._is_open_palm_robust(coords, fingers_extended, extended_count, confidence, wrist_radius):
            gestures.append(
                HandGesture(
                    "open_palm",
                    confidence,
                    handedness,
                    hand_id,
                    metadata={
                        "extended_count": float(extended_count),
                        "wrist": wrist[:2].tolist(),
                        "palm_radius": float(wrist_radius),
                    },
                )
            )

        if non_thumb_extended == 0 and pinch_distance < wrist_radius * 0.4:
            gestures.append(HandGesture("fist", max(confidence, 0.9), handedness, hand_id))

        if self._is_two_finger_v(fingers_extended, coords):
            gestures.append(HandGesture("two_finger_v", 0.7, handedness, hand_id))

        if self._is_three_finger_salute(fingers_extended, coords):
            gestures.append(HandGesture("three_finger_salute", 0.9, handedness, hand_id))

        if self._is_index_only(fingers_extended, coords):
            gestures.append(HandGesture("index_only", 0.9, handedness, hand_id))

        # Check pinky_only and rock_sign with priority - pinky_only first
        # Rock sign requires BOTH thumb and pinky, so check it only if pinky_only fails
        if self._is_pinky_only(fingers_extended, coords):
            gestures.append(HandGesture("pinky_only", 0.9, handedness, hand_id))
        elif self._is_rock_sign(fingers_extended, coords):
            # Only detect rock_sign if pinky_only is not detected
            gestures.append(HandGesture("rock_sign", 0.9, handedness, hand_id))

        if self._is_thumbs_up(coords, fingers_extended):
            gestures.append(HandGesture("thumbs_up", 0.9, handedness, hand_id))

        if self._is_thumbs_down(coords, fingers_extended):
            gestures.append(HandGesture("thumbs_down", 0.95, handedness, hand_id))

        return gestures

    def _is_open_palm_robust(
        self,
        coords: np.ndarray,
        fingers_extended: Dict[str, bool],
        extended_count: int,
        confidence: float,
        palm_radius: float,
    ) -> bool:
        """Robust open palm detection with multiple criteria."""
        
        # 1. Extended fingers >= 4 (prevents three fingers from being detected as open palm)
        if extended_count < 4:
            return False
        
        # 2. Hand confidence >= 0.5
        if confidence < 0.6:
            return False
        
        # 3. Thumb check: thumb should be extended for open palm
        if not fingers_extended.get("thumb", False):
            return False
        
        # 4. Basic palm size check
        if palm_radius < 0.06:
            return False
        
        return True

    def _is_finger_extended(
        self,
        coords: np.ndarray,
        joint_indices: Tuple[int, int, int, int],
        handedness: Optional[str],
    ) -> bool:
        mcp, pip, dip, tip = joint_indices
        v1 = coords[pip] - coords[mcp]
        v2 = coords[tip] - coords[pip]
        
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return False

        alignment = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        straight_enough = alignment > 0.4  # More lenient (was 0.5)

        # More lenient vertical check
        vertical_check = coords[tip][1] < coords[pip][1] + 0.03  # Increased tolerance

        if handedness == "left":
            vertical_check = coords[tip][1] < coords[pip][1] + 0.04

        return straight_enough and vertical_check

    def _is_thumbs_up(
        self, coords: np.ndarray, fingers_extended: Dict[str, bool]
    ) -> bool:
        if any(fingers_extended[f] for f in ("index", "middle", "ring", "pinky")):
            return False

        thumb_tip = coords[FINGER_JOINTS["thumb"][3]]
        thumb_ip = coords[FINGER_JOINTS["thumb"][2]]
        wrist = coords[0]

        thumb_vector = thumb_tip - thumb_ip
        if np.linalg.norm(thumb_vector) < 1e-6:
            return False

        pointing_up = thumb_tip[1] < wrist[1] - 0.05
        aligned = thumb_vector[1] < -0.01

        return pointing_up and aligned

    def _is_thumbs_down(
        self, coords: np.ndarray, fingers_extended: Dict[str, bool]
    ) -> bool:
        if any(fingers_extended[f] for f in ("index", "middle", "ring", "pinky")):
            return False

        thumb_tip = coords[FINGER_JOINTS["thumb"][3]]
        thumb_ip = coords[FINGER_JOINTS["thumb"][2]]
        wrist = coords[0]

        thumb_vector = thumb_tip - thumb_ip
        if np.linalg.norm(thumb_vector) < 1e-6:
            return False

        pointing_down = thumb_tip[1] > wrist[1] + 0.05
        aligned = thumb_vector[1] > 0.01

        return pointing_down and aligned

    def _is_two_finger_v(
        self, fingers_extended: Dict[str, bool], coords: np.ndarray
    ) -> bool:
        if not (
            fingers_extended["index"]
            and fingers_extended["middle"]
            and not fingers_extended["ring"]
            and not fingers_extended["pinky"]
        ):
            return False

        index_tip = coords[FINGER_JOINTS["index"][3]]
        middle_tip = coords[FINGER_JOINTS["middle"][3]]
        index_base = coords[FINGER_JOINTS["index"][0]]
        middle_base = coords[FINGER_JOINTS["middle"][0]]

        tip_separation = float(np.linalg.norm(index_tip[:2] - middle_tip[:2]))
        base_separation = float(np.linalg.norm(index_base[:2] - middle_base[:2]))

        return tip_separation > 0.08 and tip_separation > base_separation * 0.9

    def _is_three_finger_salute(
        self, fingers_extended: Dict[str, bool], coords: np.ndarray
    ) -> bool:
        """Detect three-finger salute: index, middle, ring extended; thumb and pinky not extended."""
        if not (
            fingers_extended["index"]
            and fingers_extended["middle"]
            and fingers_extended["ring"]
            and not fingers_extended["pinky"]
        ):
            return False

        if fingers_extended["thumb"]:
            return False

        index_tip = coords[FINGER_JOINTS["index"][3]]
        middle_tip = coords[FINGER_JOINTS["middle"][3]]
        ring_tip = coords[FINGER_JOINTS["ring"][3]]

        index_middle_sep = float(np.linalg.norm(index_tip[:2] - middle_tip[:2]))
        middle_ring_sep = float(np.linalg.norm(middle_tip[:2] - ring_tip[:2]))

        return index_middle_sep > 0.06 and middle_ring_sep > 0.06

    def _is_index_only(
        self, fingers_extended: Dict[str, bool], coords: np.ndarray
    ) -> bool:
        """Detect index-only gesture for zoom in."""
        if not fingers_extended["index"]:
            return False
        
        if any(fingers_extended[f] for f in ("middle", "ring", "pinky")):
            return False
        
        if fingers_extended["thumb"]:
            return False
        
        index_tip = coords[FINGER_JOINTS["index"][3]]
        index_pip = coords[FINGER_JOINTS["index"][2]]
        
        index_extended = index_tip[1] < index_pip[1]
        
        middle_tip = coords[FINGER_JOINTS["middle"][3]]
        middle_pip = coords[FINGER_JOINTS["middle"][2]]
        middle_folded = middle_tip[1] >= middle_pip[1] - 0.02
        
        return index_extended and middle_folded

    def _is_pinky_only(
        self, fingers_extended: Dict[str, bool], coords: np.ndarray
    ) -> bool:
        """Detect pinky-only gesture for zoom out."""
        if not fingers_extended["pinky"]:
            return False
        
        if any(fingers_extended[f] for f in ("index", "middle", "ring")):
            return False
        
        if fingers_extended["thumb"]:
            return False
        
        pinky_tip = coords[FINGER_JOINTS["pinky"][3]]
        pinky_pip = coords[FINGER_JOINTS["pinky"][2]]
        
        pinky_extended = pinky_tip[1] < pinky_pip[1]
        
        ring_tip = coords[FINGER_JOINTS["ring"][3]]
        ring_pip = coords[FINGER_JOINTS["ring"][2]]
        ring_folded = ring_tip[1] >= ring_pip[1] - 0.02
        
        return pinky_extended and ring_folded

    def _is_rock_sign(
        self, fingers_extended: Dict[str, bool], coords: np.ndarray
    ) -> bool:
        """Detect rock sign: thumb and pinky extended, index/middle/ring folded.
        Must be strict to avoid conflicts with pinky_only."""
        # BOTH thumb and pinky MUST be extended (strict requirement)
        if not fingers_extended.get("thumb", False) or not fingers_extended.get("pinky", False):
            return False
        
        # Index, middle, and ring MUST be folded (strict requirement)
        if any(fingers_extended.get(f, False) for f in ("index", "middle", "ring")):
            return False
        
        # Verify thumb is clearly extended
        thumb_tip = coords[FINGER_JOINTS["thumb"][3]]
        thumb_pip = coords[FINGER_JOINTS["thumb"][2]]
        thumb_extended = thumb_tip[1] < thumb_pip[1]
        
        # Verify pinky is clearly extended
        pinky_tip = coords[FINGER_JOINTS["pinky"][3]]
        pinky_pip = coords[FINGER_JOINTS["pinky"][2]]
        pinky_extended = pinky_tip[1] < pinky_pip[1]
        
        # Additional validation: thumb should be significantly extended (not just slightly)
        # This helps distinguish from cases where thumb is barely extended
        thumb_extension_amount = thumb_pip[1] - thumb_tip[1]  # Positive if extended upward
        min_thumb_extension = 0.03  # Minimum extension threshold
        
        # Additional validation: pinky should be clearly separated from ring finger
        ring_tip = coords[FINGER_JOINTS["ring"][3]]
        pinky_ring_separation = float(np.linalg.norm(pinky_tip[:2] - ring_tip[:2]))
        min_separation = 0.05  # Minimum separation to ensure pinky is clearly extended
        
        # Verify ring finger is folded
        ring_pip = coords[FINGER_JOINTS["ring"][2]]
        ring_folded = ring_tip[1] >= ring_pip[1] - 0.02
        
        return (thumb_extended and pinky_extended and 
                thumb_extension_amount >= min_thumb_extension and
                pinky_ring_separation >= min_separation and
                ring_folded)

    def _detect_swipe(
        self, key: str, coords: np.ndarray, timestamp_ms: int
    ) -> Optional[str]:
        history = self._hand_history[key]
        wrist_x = float(coords[0][0])
        history.append((timestamp_ms, wrist_x))

        while history and timestamp_ms - history[0][0] > 500:
            history.popleft()

        if len(history) < 2:
            return None

        start_time, start_x = history[0]
        end_time, end_x = history[-1]
        delta_x = end_x - start_x
        duration = end_time - start_time

        if duration < 80:
            return None

        if abs(delta_x) > 0.18 and timestamp_ms - self._last_swipe_timestamp > 500:
            self._last_swipe_timestamp = timestamp_ms
            history.clear()
            return "swipe_right" if delta_x > 0 else "swipe_left"

        return None

    def detect_two_hand_gesture(self, gestures: List[HandGesture]) -> Optional[str]:
        """Detect gestures that require BOTH hands.
        
        In EDIT mode:
        - two_open_palm = SAVE
        - two_fist = DISCARD
        
        Requires exactly 2 distinct hands showing the same gesture with high confidence.
        """
        if len(gestures) < 2:
            return None
        
        # Group gestures by hand_id to ensure we have 2 distinct hands
        gestures_by_hand: Dict[str, List[HandGesture]] = {}
        for g in gestures:
            if g.hand_id:
                if g.hand_id not in gestures_by_hand:
                    gestures_by_hand[g.hand_id] = []
                gestures_by_hand[g.hand_id].append(g)
        
        # Must have exactly 2 distinct hands
        if len(gestures_by_hand) != 2:
            return None
        
        # Get gesture lists for both hands
        hand_gesture_lists = [gestures_by_hand[hand_id] for hand_id in gestures_by_hand.keys()]
        
        # Extract gesture names and check confidence
        hand1_gestures = [g.name for g in hand_gesture_lists[0]]
        hand2_gestures = [g.name for g in hand_gesture_lists[1]]
        
        # Get highest confidence for each gesture type per hand
        hand1_open_palm_conf = max([g.confidence for g in hand_gesture_lists[0] if g.name == "open_palm"], default=0.0)
        hand2_open_palm_conf = max([g.confidence for g in hand_gesture_lists[1] if g.name == "open_palm"], default=0.0)
        
        hand1_fist_conf = max([g.confidence for g in hand_gesture_lists[0] if g.name == "fist"], default=0.0)
        hand2_fist_conf = max([g.confidence for g in hand_gesture_lists[1] if g.name == "fist"], default=0.0)
        
        # Check for two open palms (both hands must show open_palm with good confidence)
        if "open_palm" in hand1_gestures and "open_palm" in hand2_gestures:
            if hand1_open_palm_conf >= 0.5 and hand2_open_palm_conf >= 0.5:
                return "two_open_palm"  # Both hands open = SAVE (in edit mode)
        
        # Check for two fists (both hands must show fist with high confidence)
        # Require higher confidence for fist to reduce false positives
        if "fist" in hand1_gestures and "fist" in hand2_gestures:
            if hand1_fist_conf >= 0.85 and hand2_fist_conf >= 0.85:
                return "two_fist"  # Both hands fist = DISCARD (in edit mode)
        
        return None

    def get_landmarks(self) -> List[Tuple[str, np.ndarray]]:
        """Return the most recent hand landmarks keyed by hand identifier."""
        return [
            (hand_id, coords.copy())
            for hand_id, coords in zip(self._last_landmark_ids, self._last_landmarks)
        ]

    def close(self) -> None:
        if self._landmarker:
            self._landmarker.close()
