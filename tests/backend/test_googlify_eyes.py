import unittest
import numpy as np
import random
from googly_eyes.backend.lib.GooglyEyes import GooglifyEyes
from googly_eyes.backend.lib.utils.io_utils import load_config


class TestGooglifyEyes(unittest.TestCase):

    def setUp(self):
        # Set up any common configuration or mocks needed for your tests
        self.params = load_config()
        self.googlify_eyes = GooglifyEyes(self.params)

    def test_set_googly_image(self):
        with self.assertRaises(FileNotFoundError):
            self.params['googly_eye_settings']['path'] = 'image_file_does_not_exist.png'
            GooglifyEyes(self.params)

    def test_extract_face_landmarks(self):
        # Black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # No faces detected so landmarks list will be empty
        landmarks = self.googlify_eyes.extract_face_landmarks(image)
        self.assertIsInstance(landmarks, np.ndarray)
        self.assertTupleEqual(landmarks.shape, (0, 468, 2))

    def test_extract_eyes(self):
        # Black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # No faces -> eyes list will be empty
        eyes = self.googlify_eyes.extract_eyes(image)
        self.assertIsInstance(eyes, list)
        self.assertEqual(len(eyes), 0)

    def test_compute_centre_and_size(self):
        # sample eye corners (left and right) of one eye
        eye_corners = np.array([[10, 10], [20, 20]])
        random.seed(10)
        rand_eye_centre, rand_eye_size = self.googlify_eyes.randomise_eye_centre_and_size(eye_corners)
        self.assertTupleEqual(rand_eye_centre, (14, 15))
        self.assertEqual(rand_eye_size, 34)

    def test_googlify_eyes(self):
        # Black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        eyes = [((50, 50), 20)]
        result = self.googlify_eyes.googlify_eyes(image, eyes)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[-1], 4)  # googlification makes it 4 channeled image
        self.assertTupleEqual(image.shape[:2], result.shape[:2])

    def test_googlify_eyes_with_empty_eyes_list(self):
        # Black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        eyes = []
        result = self.googlify_eyes.googlify_eyes(image, eyes)
        self.assertIsInstance(result, np.ndarray)
        self.assertTupleEqual(image.shape, result.shape)
        np.testing.assert_array_equal(image, result)


if __name__ == '__main__':
    unittest.main()
