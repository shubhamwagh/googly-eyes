import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseObjectInfo:
    """
    Base data class for storing information about an object.

    Attributes:
        height (int): Height of the object.
        width (int): Width of the object.
        channels (int): Number of channels in the object.

    """
    height: int
    width: int
    channels: int


class ImageInfo(BaseObjectInfo):
    """
    Data class for storing information about an image.

    Inherits from BaseObjectInfo.

    """
    pass


class ModelInputInfo(BaseObjectInfo):
    """
    Data class for storing information about the model input.

    Inherits from BaseObjectInfo.

    """
    pass


@dataclass
class RoI:
    """
    Data class for storing information about a Region of Interest (RoI).

    Attributes:
        ORIGIN_X (int): X-coordinate of the origin of the RoI.
        ORIGIN_Y (int): Y-coordinate of the origin of the RoI.
        WIDTH (int): Width of the RoI.
        HEIGHT (int): Height of the RoI.

    """
    ORIGIN_X: int
    ORIGIN_Y: int
    WIDTH: int
    HEIGHT: int


def pad_image(image: np.ndarray) -> Tuple[np.ndarray, RoI]:
    """
    Pads an input image to make it square while preserving its aspect ratio.
    Depending on the original image's aspect ratio padding is either done on the bottom or on right

    Args:
        - image (np.ndarray): Input image, can be grayscale or color.

    Returns:
        Tuple[np.ndarray, RoI]:
        A tuple containing the padded image and the Region of Interest (RoI) represented
        by a RoI data class with attributes ORIGIN_X, ORIGIN_Y, WIDTH, and HEIGHT.
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    aspect_ratio = width / height

    if aspect_ratio > 1.0:
        delta = width - height
        img_pad = cv2.copyMakeBorder(image, 0, delta, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        delta = height - width
        img_pad = cv2.copyMakeBorder(image, 0, 0, 0, delta, cv2.BORDER_CONSTANT, (0, 0, 0))
    roi = RoI(ORIGIN_X=0, ORIGIN_Y=0, WIDTH=img_pad.shape[1], HEIGHT=img_pad.shape[0])
    return img_pad, roi
