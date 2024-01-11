import os
import cv2
import numpy as np
import random

from .googlyeyes_constants import LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS
from googly_eyes.backend.lib.BlazeFace import BlazeFaceDetector
from googly_eyes.backend.lib.FaceMesh import FaceMeshLandmarks
from googly_eyes.backend.lib.utils import image_utils

from typing import Tuple, List, Dict, Union

EyeCentreAndSize = Tuple[Tuple[int, int], int]


class GooglifyEyes:
    def __init__(self, params: Dict[str, Union[Dict[str, Union[float, str]], float, str]]):
        """
        Args:
            params (dict): Parameters for GooglifyEyes.
        """
        self.params = params
        self.googly_filter = self.set_googly_image()
        face_detection_settings = params['face_detection_settings']
        self.face_detector = BlazeFaceDetector(score_threshold=face_detection_settings.get('score_threshold', 0.7),
                                               iou_threshold=face_detection_settings.get('iou_threshold', 0.3))
        self.face_mesh = FaceMeshLandmarks()

    def generate(self, img: np.ndarray) -> np.ndarray:
        """
        Generates a googlified version of the input image with googly eyes.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Googlified image with googly eyes.
        """
        eyes = self.extract_eyes(img)
        return self.googlify_eyes(img, eyes)

    def set_googly_image(self) -> np.ndarray:
        """
        Sets the googly eye image.

        Raises:
            FileNotFoundError

        Returns:
            np.ndarray: Googly eye image.
        """
        googly_eye_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           "../../../../", self.params['googly_eye_settings']['path'])
        if os.path.isfile(googly_eye_img_path):
            googly_filter = cv2.imread(googly_eye_img_path,
                                       cv2.IMREAD_UNCHANGED)
            return googly_filter
        raise FileNotFoundError(f'{googly_eye_img_path} file does not exist! Please give correct image file path')

    def extract_face_landmarks(self, img: np.ndarray) -> np.ndarray:
        """
        Extracts face landmarks from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: 468 facial landmarks.
        """
        # make it square shaped image -> ensures squared face bounding box
        pad_img, roi = image_utils.pad_image(img)
        # detect all faces in the image
        detection_results = self.face_detector.detect_faces(pad_img)
        # extract face landmarks for all detected faces
        face_landmarks = self.face_mesh.compute_2d_landmarks(pad_img, detection_results)
        return face_landmarks

    def extract_eyes(self, img: np.ndarray) -> List[EyeCentreAndSize]:
        """
        Extracts eye centers and sizes from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            List[EyeCentreAndSize]: List of eye centers and sizes.
        """
        face_landmarks = self.extract_face_landmarks(img)
        eyes = []
        for face in face_landmarks:
            left_eye_corners = face[LEFT_EYE_CORNERS]
            right_eye_corners = face[RIGHT_EYE_CORNERS]
            eyes.extend(
                [self.randomise_eye_centre_and_size(left_eye_corners),
                 self.randomise_eye_centre_and_size(right_eye_corners)])
        return eyes

    def googlify_eyes(self, img: np.ndarray, eyes: List[EyeCentreAndSize]) -> np.ndarray:
        """
        Googlifies the eyes in the input image.

        Args:
            img (np.ndarray): Input image.
            eyes (List[EyeCentreAndSize]): List of eye centers and sizes.

        Returns:
            np.ndarray: Googlified image with googly eyes.
        """
        if len(eyes) == 0: return img

        # add alpha channel for transparency computation -> BGRA format
        img = np.concatenate((img, np.ones(img.shape[:2], img.dtype)[..., np.newaxis] * 255), axis=-1)

        for eye_centre, eye_size in eyes:
            x, y = eye_centre
            if eye_size % 2 == 1:
                eye_size += 1
            half_size = eye_size // 2

            # extract eye roi from the original image
            roi = img[y - half_size: y + half_size, x - half_size: x + half_size]

            # resize googly eye filter -> same size as roi
            resized_googly_filter = cv2.resize(self.googly_filter, (eye_size, eye_size), interpolation=cv2.INTER_AREA)

            assert roi.shape == resized_googly_filter.shape

            # create filter masks for blending
            googly_eye_mask = (resized_googly_filter[..., 3] != 0).astype(np.uint8) * 255
            google_eye_bg_mask = cv2.bitwise_not(googly_eye_mask)
            googly_eye = cv2.bitwise_and(resized_googly_filter, resized_googly_filter, mask=googly_eye_mask)
            googly_eye_background = cv2.bitwise_and(roi, roi, mask=google_eye_bg_mask)

            # random rotate googly_eye image
            rot_mat = cv2.getRotationMatrix2D((half_size, half_size), random.randrange(360), 1)
            rotated_googly_eye = cv2.warpAffine(googly_eye, rot_mat, (eye_size, eye_size))

            # linear blending
            blended_googly_eye = cv2.add(googly_eye_background, rotated_googly_eye)
            # replace with the blended googly eye image
            img[y - half_size: y + half_size, x - half_size: x + half_size] = blended_googly_eye
        return img

    def randomise_eye_centre_and_size(self, eye_corners: np.ndarray) -> EyeCentreAndSize:
        """
        Randomizes eye center and size.

        Args:
            eye_corners (np.ndarray): Eye corners.

        Returns:
            EyeCentreAndSize: Randomized eye center and size.
        """
        assert len(eye_corners) == 2
        eye_dist = int(np.linalg.norm(eye_corners[0] - eye_corners[1]))
        eye_centre = eye_corners.mean(axis=0)

        # random_scale
        settings = self.params.get('googly_eye_settings',
                                   {'size_multiplier': 2.0, 'size_inc_percent': 0.4, 'centre_offset_percent': 0.3})
        eye_dist_scaled = int(eye_dist * settings.get('size_multiplier', 2.0))
        eye_dist_scaled += int(eye_dist_scaled * random.uniform(0, settings.get('size_inc_percent', 0.4)))

        # random centre offset
        centre_offset_percent = settings.get('centre_offset_percent', 0.3)
        random_offset_x = eye_dist * random.uniform(-centre_offset_percent,
                                                    centre_offset_percent)
        random_offset_y = eye_dist * random.uniform(-centre_offset_percent,
                                                    centre_offset_percent)
        eye_centre[0] += random_offset_x
        eye_centre[1] += random_offset_y
        eye_centre_w_offset = tuple(eye_centre.astype(np.int32).tolist())

        return eye_centre_w_offset, eye_dist_scaled
