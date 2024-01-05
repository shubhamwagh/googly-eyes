import unittest
import numpy as np
from googly_eyes.backend.lib.utils.image_utils import pad_image, RoI


class TestImageUtils(unittest.TestCase):

    def test_pad_right_image(self):
        # sample image
        image = np.ones((100, 80), dtype=np.uint8) * 255  # White image, 80x100
        padded_image, roi = pad_image(image)

        # check padded image shape
        self.assertEqual(padded_image.shape, (100, 100))

        # ensure RoI is correct
        expected_roi = RoI(ORIGIN_X=0, ORIGIN_Y=0, WIDTH=100, HEIGHT=100)
        self.assertEqual(roi, expected_roi)

        # Check right border is black
        self.assertEqual(np.sum(padded_image[:, -20:]), 0)

    def test_pad_bottom_image(self):
        # sample image
        image = np.ones((80, 100), dtype=np.uint8) * 255  # White image, 100x80
        padded_image, roi = pad_image(image)

        # check padded image shape
        self.assertEqual(padded_image.shape, (100, 100))

        # ensure RoI is correct
        expected_roi = RoI(ORIGIN_X=0, ORIGIN_Y=0, WIDTH=100, HEIGHT=100)
        self.assertEqual(roi, expected_roi)

        # Check bottom border is black
        self.assertEqual(np.sum(padded_image[-20:, :]), 0)


if __name__ == '__main__':
    unittest.main()
