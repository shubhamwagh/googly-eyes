import os
import cv2
import unittest
import numpy as np
import tensorflow as tf
from googly_eyes.backend.lib.FaceMesh import FaceMeshLandmarks


class TestFaceMeshLandmarks(unittest.TestCase):
    def setUp(self):
        self.face_mesh = FaceMeshLandmarks()
        sample_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/misc/girl.jpg")
        self.cv_img = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)
        self.blazeface_detections = np.array([[0.24812626, 0.34753296, 0.53638122, 0.63578799, 0.43028343,
                                               0.33029138, 0.5417786, 0.32483621, 0.48258972, 0.402166,
                                               0.48696939, 0.46024916, 0.37805289, 0.35793077, 0.61235455,
                                               0.3481114, 0.81457238]])

    def test_pre_process_img_for_inference(self):
        face_crops, affine, box = self.face_mesh.pre_process_img_for_inference(self.cv_img, self.blazeface_detections)

        # ensure one face
        self.assertEqual(face_crops.shape[0], 1)
        self.assertEqual(box.shape[0], 1)
        self.assertEqual(affine.shape[0], 1)

        # check affine matrix and face box values
        expected_affine = np.array([[[1.3052081e+00, 4.2531796e-02, 1.2154601e+02],
                                     [-4.2531811e-02, 1.3052082e+00, 1.2388429e+01]]], dtype=np.float32)

        expected_box = np.array([[[121.54601627, 129.66959308, 370.84077047, 378.96434728],
                                  [12.38842916, 261.68318336, 4.26485236, 253.55960656]]])

        np.testing.assert_array_almost_equal(affine, expected_affine)
        np.testing.assert_array_almost_equal(box, expected_box)

        # ensure face_crops is normalised [-1.0, 1.0]
        self.assertLessEqual(np.max(face_crops), 1.0)
        self.assertGreaterEqual(np.min(face_crops), -1.0)

    def test_inference(self):
        # Mock input tensor
        input_tensor = tf.constant(np.random.uniform(-1.0, 1.0, size=(1, 192, 192, 3)).astype(np.float32))
        # inference result
        raw_landmarks, raw_conf_scores = self.face_mesh.inference(input_tensor)

        num_facial_landmarks, dim = 468, 3
        self.assertTupleEqual(raw_landmarks.shape, (num_facial_landmarks * dim,))
        self.assertEqual(raw_conf_scores.ndim, 0)

    def test_compute_2d_landmarks_complete_pipeline(self):
        landmarks = self.face_mesh.compute_2d_landmarks(self.cv_img, self.blazeface_detections)

        # ensure one face
        self.assertEqual(landmarks.shape[0], 1)
        self.assertTupleEqual(landmarks.shape[1:], (468, 2))  # -> (468, 2) facial landmarks

        # Ensure all landmark coordinates are within the image size
        img_height, img_width, _ = self.cv_img.shape
        self.assertTrue(
            np.all((landmarks >= 0) & (landmarks[:, 0] < img_width) & (landmarks[:, 1] >= 0) & (
                    landmarks[:, 1] < img_height))
        )


if __name__ == '__main__':
    unittest.main()
