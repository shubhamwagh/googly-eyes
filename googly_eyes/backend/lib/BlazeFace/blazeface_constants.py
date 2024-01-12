from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyPoints:
    Y_MIN = 0
    X_MIN = 1
    Y_MAX = 2
    X_MAX = 3

    FIRST_X_COORD_INDEX = 4
    LAST_X_COORD_INDEX = 14
    CONFIDENCE_SCORE_INDEX = 16

    LENGTH = 17


@dataclass(frozen=True)
class AnchorOptions:
    """
    Attributes for blaze face model
    More info - https://github.com/google/mediapipe/blob/0.8.3.2/mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
    Meaning of each attrbute - https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.proto
    """
    NUM_LAYERS: int = 4
    MIN_SCALE: float = 0.15625
    MAX_SCALE: float = 0.75
    INPUT_SIZE_HEIGHT: int = 256
    INPUT_SIZE_WIDTH: int = 256
    ANCHOR_OFFSET_X: float = 0.5
    ANCHOR_OFFSET_Y: float = 0.5
    STRIDES: Tuple[int, int, int, int] = (16, 32, 32, 32)
    ASPECT_RATIOS: Tuple[float] = (1.0,)
    REDUCE_BOXES_IN_LOWEST_LAYER: bool = False
    INTERPOLATED_SCALE_ASPECT_RATIO: float = 1.0
    FIXED_ANCHOR_SIZE: bool = True
