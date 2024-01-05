import os
import cv2
import unittest
import numpy as np
import tensorflow as tf
from googly_eyes.backend.lib.BlazeFace import BlazeFaceDetector


class TestBlazeFaceDetector(unittest.TestCase):

    def setUp(self):
        score_threshold = 0.7
        iou_threshold = 0.3
        self.detector = BlazeFaceDetector(score_threshold=score_threshold,
                                          iou_threshold=iou_threshold)
        self.sample_img = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/misc/multi_face.png")

    def test_pre_process_img_for_inference(self):
        # Mock image data
        image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        # Call the method
        result = self.detector.pre_process_img_for_inference(image)
        # Assert the result is a tf.Tensor
        self.assertIsInstance(result, tf.Tensor)
        self.assertAlmostEqual(np.max(result), 0.99, places=2)
        self.assertAlmostEqual(np.min(result), -1.0, places=2)
        self.assertListEqual(np.shape(result).as_list(), [1, 256, 256, 3])

    def test_inference_and_tensors_to_detections(self):
        # Mock input tensor with no face
        input_tensor = tf.constant(np.random.uniform(-1.0, 1.0, size=(1, 256, 256, 3)).astype(np.float32))
        # inference result
        raw_box_values, raw_detection_scores = self.detector.inference(input_tensor)

        self.assertIsInstance(raw_box_values, np.ndarray)
        self.assertIsInstance(raw_detection_scores, np.ndarray)
        self.assertTupleEqual(raw_box_values.shape, (896, 16))
        self.assertTupleEqual(raw_detection_scores.shape, (896, 1))

        # tensor_to_detections
        result = self.detector.tensors_to_detections(raw_box_values, raw_detection_scores)
        # Assert the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        # since it is mock input so no faces
        self.assertTupleEqual(result.shape, (0, 17))

    def test_intersect(self):
        box_a = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
        box_b = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])
        result = self.detector.intersect(box_a, box_b)
        expected_result = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(result, expected_result)

    def test_jaccard(self):
        box_a = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
        box_b = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])
        result = self.detector.jaccard(box_a, box_b)
        expected_result = np.array([[1 / 7, 1 / 7], [1 / 7, 1 / 7]])
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_weighted_non_max_suppression(self):
        # empty detections
        detections = np.zeros((0, 17))
        result = self.detector.weighted_non_max_suppression(detections)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_detect_faces_complete_pipeline(self):
        img = cv2.imread(self.sample_img, cv2.IMREAD_COLOR)
        result = self.detector.detect_faces(img)
        self.assertIsInstance(result, np.ndarray)
        # 8 faces and 17 dim
        self.assertTupleEqual(result.shape, (8, 17))


if __name__ == '__main__':
    unittest.main()
