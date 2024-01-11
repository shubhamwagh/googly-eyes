import os
import cv2
import numpy as np
import tensorflow as tf

from .facemesh_utils import denormalise_detections, draw_roi, draw_landmarks
from googly_eyes.backend.lib.TFLite import TFLite
from googly_eyes.backend.lib.utils import image_utils

from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "../../models/facemesh/face_landmark.tflite")

# mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
LEFT_EYE = 0  # Left eye index on the viewer's left-hand side
RIGHT_EYE = 1  # Right eye index on the viewer's right-hand side
THETA_0 = 0  # initial theta value of the box
D_SCALE = 1.7  # Scale value for output mesh.
Dy = 0.

NUM_FACIAL_LANDMARKS = 468  # number of face landmarks
NUM_DIM = 3


class FaceMeshLandmarks(TFLite):
    def __init__(self):
        super(FaceMeshLandmarks, self).__init__(model_path=MODEL_PATH)

    def compute_raw_landmarks(self, image: np.ndarray, blazeface_detections: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute raw facial landmarks and related information based on the input image and BlazeFace detections.

        Args:
            image (np.ndarray): Input image.
            blazeface_detections (np.ndarray): BlazeFace detections.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - boxes (np.ndarray): Oriented Bounding box coordinates.
            - landmarks (np.ndarray): Facial landmarks.
            - confidences (np.ndarray): Confidence scores.
            - affines (np.ndarray): Affine transformation matrices.
        """
        num_of_detections = blazeface_detections.shape[0]
        imgs, affines, boxes = self.pre_process_img_for_inference(image, blazeface_detections)

        # Adjust matrix dimensions
        _, input_height, input_width, input_channels = self.model_input_details[0]['shape']
        assert input_height == input_width

        landmarks = np.zeros((num_of_detections, NUM_FACIAL_LANDMARKS, NUM_DIM))
        confidences = np.zeros((num_of_detections, 1))
        for ind in range(imgs.shape[0]):
            tensor = tf.convert_to_tensor(imgs[ind].reshape(1, input_height, input_width, input_channels),
                                          dtype=tf.float32)
            landmark_points, raw_confidence_score = self.inference(tensor)
            landmarks[ind, :, :] = landmark_points.reshape(NUM_FACIAL_LANDMARKS, NUM_DIM)
            confidences[ind, :] = 1.0 / (1.0 + np.exp(-raw_confidence_score))

        return boxes, landmarks, confidences, affines

    def compute_2d_landmarks(self, image: np.ndarray, blazeface_detections: np.ndarray) -> np.ndarray:
        """
        Compute 2D facial landmarks based on the input image and BlazeFace detections.

        Args:
            image (np.ndarray): Input image.
            blazeface_detections (np.ndarray): BlazeFace detections.

        Returns:
            np.ndarray: 2D facial landmarks.
        """
        boxes, landmarks, confidences, affines = self.compute_raw_landmarks(image, blazeface_detections)
        return self.denormalise_all_face_landmarks(landmarks, affines)

    def inference(self, input_tensor: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on the input tensor using the model.

        Args:
            input_tensor (tf.Tensor): Input tensor for inference.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - landmarks (np.ndarray): Predicted facial landmarks.
            - confidence (np.ndarray): Predicted confidence scores.
        """
        self.interpreter.set_tensor(self.model_input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        landmarks = self.interpreter.get_tensor(self.model_output_details[0]['index'])
        confidence = self.interpreter.get_tensor(self.model_output_details[1]['index'])
        return np.squeeze(landmarks), np.squeeze(confidence)

    def pre_process_img_for_inference(self, image: np.ndarray, blazeface_detections: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pre-process the input image for inference.

        Args:
            image (np.ndarray): Input image.
            blazeface_detections (np.ndarray): BlazeFace detections.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - img (np.ndarray): Extracted ROIs
            - affine (np.ndarray): Affine transformation matrix.
            - box (np.ndarray): Oriented Bounding box coordinates.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = img.shape
        self.image_info = image_utils.ImageInfo(height=img_height, width=img_width, channels=img_channels)

        denormalised_detections = denormalise_detections(blazeface_detections, (img_height, img_width))
        xc, yc, scale, theta = self.detection2roi(denormalised_detections, detection2roi_method='box')
        img, affine, box = self.extract_roi(img, xc, yc, theta, scale)
        return img, affine, box

    @staticmethod
    def detection2roi(detection: np.ndarray, detection2roi_method: str = 'box') -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert detections from detector to an oriented bounding box.
        Adapted from:
        mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
        The center and size of the box is calculated from the center
        of the detected box. Rotation is calculated from the vector
        between LEFT_EYE and RIGHT_EYE relative to theta0. The box is scaled
        and shifted by dscale and dy.

        Args:
            detection (np.ndarray): Detections from the detector.
            detection2roi_method (str, optional): Method for converting detections to an oriented bounding box. Defaults to 'box'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - xc (np.ndarray): X-coordinate of the box center.
            - yc (np.ndarray): Y-coordinate of the box center.
            - scale (np.ndarray): Scale factor of the box.
            - theta (np.ndarray): Rotation angle of the box.
        """
        if detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:, 1] + detection[:, 3]) / 2
            yc = (detection[:, 0] + detection[:, 2]) / 2
            scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

        elif detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            x1 = detection[:, 4 + 2 * LEFT_EYE]
            y1 = detection[:, 4 + 2 * LEFT_EYE + 1]
            xc = detection[:, 4 + 2 * RIGHT_EYE]
            yc = detection[:, 4 + 2 * RIGHT_EYE + 1]
            scale = np.sqrt(((xc - x1) ** 2 + (yc - y1) ** 2)) * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported" % detection2roi_method)

        yc += Dy * scale
        scale *= D_SCALE

        # compute box rotation
        x1 = detection[:, 4 + 2 * LEFT_EYE]
        y1 = detection[:, 4 + 2 * LEFT_EYE + 1]
        x0 = detection[:, 4 + 2 * RIGHT_EYE]
        y0 = detection[:, 4 + 2 * RIGHT_EYE + 1]
        theta = np.arctan2(y0 - y1, x0 - x1) - THETA_0
        return xc, yc, scale, theta

    def extract_roi(self, frame: np.ndarray, xc: np.ndarray, yc: np.ndarray, theta: np.ndarray, scale: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts a Region of Interest (ROI) from the given frame based on specified parameters.

        Parameters:
        - frame (np.ndarray): Input image frame.
        - xc (np.ndarray): X-coordinates of the center of the ROI.
        - yc (np.ndarray): Y-coordinates of the center of the ROI.
        - theta (np.ndarray): Rotation angles of the ROI.
        - scale (np.ndarray): Scale factors for the ROI.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - imgs (np.ndarray): Extracted ROIs.
        - affines (np.ndarray): Affine transformation matrices for each ROI.
        - points (np.ndarray): Oriented Bounding box coordinates
        """

        # take points on unit square and transform them according to the roi
        points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]]).reshape(1, 2, 4)
        points = points * scale.reshape(-1, 1, 1) / 2
        theta = theta.reshape(-1, 1, 1)
        R = np.concatenate((
            np.concatenate((np.cos(theta), -np.sin(theta)), 2),
            np.concatenate((np.sin(theta), np.cos(theta)), 2),
        ), 1)
        center = np.concatenate((xc.reshape(-1, 1, 1), yc.reshape(-1, 1, 1)), 1)
        points = R @ points + center

        # use the points to compute the affine transform that maps
        # these points back to the output square
        height, width = self.model_input_info.height, self.model_input_info.width
        # only 3 points needed to compute affine transformation
        # Ensure correspondence and order of point is similar.
        points1 = np.array([[0, 0, width - 1],
                            [0, height - 1, 0]], dtype='float32').T
        affines = []
        imgs = []
        for i in range(points.shape[0]):
            pts = points[i, :, :3].T.astype('float32')
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (width, height), borderValue=0.0)
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affines.append(affine)
        if imgs:
            imgs = np.stack(imgs).astype('float32') / 127.5 - 1.0
            affines = np.stack(affines)
        else:
            imgs = np.zeros((0, height, width, 3))
            affines = np.zeros((0, 2, 3))

        return imgs, affines, points

    def denormalise_per_face_landmarks(self, landmark: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """
        Denormalizes facial landmarks based on the given affine transformation matrix.

        Parameters:
        - landmark (np.ndarray): Normalized facial landmarks.
        - affine (np.ndarray): Affine transformation matrix.

        Returns:
        np.ndarray: Denormalized facial landmarks.
        """
        landmark = affine @ np.vstack((landmark[:, :2].T, np.ones((1, landmark.shape[0]))))
        return landmark.T

    def denormalise_all_face_landmarks(self, landmarks: np.ndarray, affines: np.ndarray) -> np.ndarray:
        """
        Denormalizes all facial landmarks based on the given affine transformation matrices.

        Parameters:
        - landmarks (np.ndarray): Normalized facial landmarks for multiple faces.
        - affines (np.ndarray): Affine transformation matrices for each face.

        Returns:
        np.ndarray: Denormalized facial landmarks for multiple faces.
        """
        N = len(landmarks)
        denorm_landmarks = np.zeros((N, NUM_FACIAL_LANDMARKS, 2))
        for ind in range(N):
            denorm_landmarks[ind] = self.denormalise_per_face_landmarks(landmarks[ind],
                                                                        affines[ind])
        return denorm_landmarks

    def draw_landmarks_and_box(self, img: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, confidences: np.ndarray,
                               affines: np.ndarray) -> np.ndarray:
        """
        Draws facial landmarks and bounding boxes on the input image.

        Parameters:
        - img (np.ndarray): Input image.
        - boxes (np.ndarray): Bounding boxes.
        - landmarks (np.ndarray): Facial landmarks.
        - confidences (np.ndarray): Confidence scores.
        - affines (np.ndarray): Affine transformation matrices.

        Returns:
        np.ndarray: Image with drawn landmarks and bounding boxes.
        """
        assert landmarks.shape[0] == boxes.shape[0]
        for ind in range(len(landmarks)):
            img = draw_roi(img, boxes[ind], confidences[ind])
            denormalised_landmarks = self.denormalise_per_face_landmarks(landmarks[ind],
                                                                         affines[ind])
            img = draw_landmarks(img, denormalised_landmarks)
        return img
