import math
import os
import cv2
import numpy as np
import tensorflow as tf

from .blazeface_utils import generate_anchors
from .blazeface_constants import KeyPoints
from googly_eyes.backend.lib.TFLite import TFLite
from googly_eyes.backend.lib.utils import image_utils
from typing import Tuple, List

KEY_POINT_SIZE = 6
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "../../models/blazeface/face_detection_back_256x256_float16_quant.tflite")


class BlazeFaceDetector(TFLite):
    """
    BlazeFaceDetector class
    """

    def __init__(self, score_threshold: float = 0.70, iou_threshold: float = 0.3) -> None:
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        super(BlazeFaceDetector, self).__init__(model_path=MODEL_PATH)

        # Generate anchors for model
        self.anchors: np.ndarray = generate_anchors()

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Detect faces in the given image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Detected faces with shape (num_faces, 17).
        """
        # Prepare image for inference
        input_tensor = self.pre_process_img_for_inference(image)

        # Perform inference on the image
        raw_box_values, raw_detection_scores = self.inference(input_tensor)

        detections = self.tensors_to_detections(raw_box_values, raw_detection_scores)

        # Non-maximum suppression to remove overlapping detections:
        faces = self.weighted_non_max_suppression(detections)

        # stack for multiple faces
        filtered_detections = np.stack(faces) if len(faces) > 0 else np.zeros((0, 17))

        # return detection_results
        return filtered_detections

    def pre_process_img_for_inference(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess the input image for model inference.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            tf.Tensor: Preprocessed image as a TensorFlow tensor.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = img.shape
        self.image_info = image_utils.ImageInfo(height=img_height, width=img_width, channels=img_channels)

        # Normalise pixel values in range [-1.0, 1.0]
        _, input_height, input_width, input_channels = self.model_input_details[0]['shape']
        img_resized = cv2.resize(img, (input_height, input_width))
        img_input = (img_resized / 127.5) - 1.0

        # Adjust matrix dimensions
        reshape_img = img_input.reshape(1, input_height, input_width, input_channels)
        tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

        return tensor

    def inference(self, input_tensor: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference using the input tensor.

        Args:
            input_tensor (tf.Tensor): Input tensor for the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - raw_box_values (np.ndarray): Matrix of 896 x 16 with information about the detected faces.
            - raw_detection_scores (np.ndarray): Matrix with the raw detection scores.
        """
        self.interpreter.set_tensor(self.model_input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Matrix with the raw detection scores
        raw_detection_scores = np.squeeze(
            np.concatenate((self.interpreter.get_tensor(self.model_output_details[0]['index']),  # 512 x 1
                            self.interpreter.get_tensor(self.model_output_details[1]['index'])),  # 384 x 1
                           axis=1)).reshape(-1, 1)

        # Matrix of 896 x 16 with information about the detected faces
        raw_box_values = np.squeeze(
            np.concatenate((self.interpreter.get_tensor(self.model_output_details[2]['index']),  # 512 x 16
                            self.interpreter.get_tensor(self.model_output_details[3]['index'])),  # 384 x 16
                           axis=1))

        return raw_box_values, raw_detection_scores

    @staticmethod
    def m_sigmoid(x):
        """
        Apply the sigmoid function to the input.

        Args:
            x (float): Input value.

        Returns:
            float: Output after applying the sigmoid function.
        """
        return 1.0 / (1.0 + math.exp(-x))

    def tensors_to_detections(self, raw_box_tensor: np.ndarray, raw_score_tensor: np.ndarray) -> np.ndarray:
        """
        The output of the neural network is a tensor of shape (896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto

        Args:
            raw_box_tensor (np.ndarray): Tensor with bounding box regressor predictions of shape (896, 16).
            raw_score_tensor (np.ndarray): Tensor with classification confidences of shape (896, 1).

        Returns:
            np.ndarray: List of (num_detections, 17) tensors, one for each image in the batch.
        """

        detection_boxes = self.decode_boxes(raw_box_tensor)

        thresh = 100.0
        raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
        detection_scores = np.vectorize(self.m_sigmoid)(raw_score_tensor)

        # stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        good_detection_index = np.where(detection_scores >= self.score_threshold)[0]

        output_detections = np.concatenate(
            (detection_boxes[good_detection_index], detection_scores[good_detection_index]), axis=-1)

        return output_detections

    def decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """
        Converts the predictions into actual coordinates using
        the anchor boxes. Processes the all 896 boxes at once.

        Args:
            raw_boxes (np.ndarray): Raw box predictions.

        Returns:
            np.ndarray: Processed boxes with actual coordinates.
        """
        x_scale = self.model_input_info.width
        y_scale = self.model_input_info.height
        w_scale = self.model_input_info.width
        h_scale = self.model_input_info.height

        boxes = np.zeros(raw_boxes.shape, raw_boxes.dtype)

        x_center = raw_boxes[..., 0] / x_scale * self.anchors[:, 2] + self.anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * self.anchors[:, 3] + self.anchors[:, 1]

        w = raw_boxes[..., 2] / w_scale * self.anchors[:, 2]
        h = raw_boxes[..., 3] / h_scale * self.anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(KEY_POINT_SIZE):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / x_scale * self.anchors[:, 2] + self.anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / y_scale * self.anchors[:, 3] + self.anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def draw_detections(self, img: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on the input image.

        Args:
            img (np.ndarray): Input image.
            results (np.ndarray): Detections results.

        Returns:
            np.ndarray: Image with drawn bounding boxes and keypoints.
        """
        for det in results:
            y1 = (det[KeyPoints.Y_MIN] * self.image_info.height).astype(int)
            x1 = (det[KeyPoints.X_MIN] * self.image_info.width).astype(int)

            x2 = (det[KeyPoints.X_MAX] * self.image_info.width).astype(int)
            y2 = (det[KeyPoints.Y_MAX] * self.image_info.height).astype(int)

            cv2.rectangle(img, (x1, y1), (x2, y2), (22, 250, 22), 2)
            cv2.putText(img, '{:.2f}'.format(det[-1]), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (22, 22, 250), 2)

            for index in range(KeyPoints.FIRST_X_COORD_INDEX, KeyPoints.LAST_X_COORD_INDEX + 1, 2):
                x_keypoint = (det[index] * self.image_info.width).astype(int)
                y_keypoint = (det[index + 1] * self.image_info.height).astype(int)
                cv2.circle(img, (x_keypoint, y_keypoint), 4, (214, 202, 18), -1)

        return img

    @staticmethod
    def intersect(box_a, box_b):
        """
        We resize both tensors to [A,B,2]:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.

        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Returns:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.shape[0]
        B = box_b.shape[0]
        max_xy = np.minimum(np.resize(np.expand_dims(box_a[:, 2:], axis=1), (A, B, 2)),
                            np.resize(np.expand_dims(box_b[:, 2:], axis=0), (A, B, 2)))
        min_xy = np.maximum(np.resize(np.expand_dims(box_a[:, :2], axis=1), (A, B, 2)),
                            np.resize(np.expand_dims(box_b[:, :2], axis=0), (A, B, 2)))
        inter = np.clip(max_xy - min_xy, a_min=0, a_max=None)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self, box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
        """
        Compute the jaccard overlap of two sets of boxes.
        The jaccard overlap is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.shape[0], box_b.shape[0]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = np.resize(np.expand_dims(((box_a[:, 2] - box_a[:, 0]) *
                                           (box_a[:, 3] - box_a[:, 1])), axis=1), inter.shape)  # [A,B]
        area_b = np.resize(np.expand_dims(((box_b[:, 2] - box_b[:, 0]) *
                                           (box_b[:, 3] - box_b[:, 1])), axis=0), inter.shape)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def overlap_similarity(self, box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
        """
        Computes the IOU between a bounding box and set of other boxes.

        Args:
            box (np.ndarray): Bounding box.
            other_boxes (np.ndarray): Set of other bounding boxes.

        Returns:
            np.ndarray: IoU values.
        """
        return np.squeeze(self.jaccard(np.expand_dims(box, axis=0), other_boxes), axis=0)

    def weighted_non_max_suppression(self, detections: np.ndarray) -> List[np.ndarray]:
        """
        The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (face_count, 17).

        Based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto

        Args:
            detections (np.ndarray): Input detections of shape (face_count, 17).

        Returns:
            List[np.ndarray]: List of weighted detections.
        """
        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = np.argsort(-detections[:, KeyPoints.CONFIDENCE_SCORE_INDEX])
        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = self.overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.iou_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = np.copy(detection)
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :KeyPoints.CONFIDENCE_SCORE_INDEX]
                scores = detections[overlapping, KeyPoints.CONFIDENCE_SCORE_INDEX:KeyPoints.CONFIDENCE_SCORE_INDEX + 1]
                total_score = np.sum(scores)
                weighted = np.sum((coordinates * scores), axis=0) / total_score
                weighted_detection[:KeyPoints.CONFIDENCE_SCORE_INDEX] = weighted
                weighted_detection[KeyPoints.CONFIDENCE_SCORE_INDEX] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections
