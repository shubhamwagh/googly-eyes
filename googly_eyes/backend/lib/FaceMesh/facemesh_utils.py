import numpy as np
import cv2
from typing import Tuple


def denormalise_detections(blazeface_detections: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Denormalize BlazeFace detections based on the input image size.

    Args:
        blazeface_detections (np.ndarray): BlazeFace detections.
        size (Tuple[int, int]): Size of the input image (height, width).

    Returns:
        np.ndarray: Denormalized BlazeFace detections.
    """
    denormalised_detections = np.zeros_like(blazeface_detections)

    height, width = size
    denormalised_detections[:, 0] = blazeface_detections[:, 0] * height
    denormalised_detections[:, 1] = blazeface_detections[:, 1] * width
    denormalised_detections[:, 2] = blazeface_detections[:, 2] * height
    denormalised_detections[:, 3] = blazeface_detections[:, 3] * width

    denormalised_detections[:, 4::2] = blazeface_detections[:, 4::2] * width
    denormalised_detections[:, 5::2] = blazeface_detections[:, 5::2] * height

    return denormalised_detections


def draw_roi(img: np.ndarray, roi: np.ndarray, confidence: np.ndarray):
    """
    Draw a region of interest (ROI) with confidence on the input image.

    Args:
        img (np.ndarray): Input image.
        roi (np.ndarray): ROI coordinates.
        confidence (np.ndarray): Confidence score.

    Returns:
        np.ndarray: Image with drawn ROI.
    """
    (x1, x2, x3, x4), (y1, y2, y3, y4) = roi
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)

    cv2.putText(img, '{:.2f}'.format(confidence[0]), (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (22, 22, 250), 2)
    return img


def draw_landmarks(img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), size: int = 2) \
        -> np.ndarray:
    """
    Draw facial landmarks on the input image.

    Args:
        img (np.ndarray): Input image.
        points (np.ndarray): Facial landmarks.
        color (Tuple[int, int, int], optional): Color of the landmarks. Defaults to (255, 0, 0).
        size (int, optional): Size of the drawn landmarks. Defaults to 2.

    Returns:
        np.ndarray: Image with drawn landmarks.
    """
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)

    return img
