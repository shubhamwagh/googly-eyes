import os
import unittest
import tensorflow as tf
from googly_eyes.backend.lib.TFLite import TFLite


class ConcreteTFLite(TFLite):
    def __init__(self, model_path: str):
        super(ConcreteTFLite, self).__init__(model_path)

    def inference(self, input_tensor: tf.Tensor) -> tf.Tensor:
        # Mock implementation for testing
        return input_tensor


class TestTFLite(unittest.TestCase):

    def setUp(self):
        self.valid_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "../../googly_eyes/backend/models/blazeface"
                                             "/face_detection_back_256x256_float16_quant.tflite")

        self.invalid_model_path = 'path/to/nonexistent/model.tflite'

    def test_initialization_with_valid_model(self):
        tflite_instance = ConcreteTFLite(self.valid_model_path)
        self.assertIsNotNone(tflite_instance.interpreter)
        self.assertIsNotNone(tflite_instance.model_input_details)
        self.assertIsNotNone(tflite_instance.model_input_info)
        self.assertIsNotNone(tflite_instance.model_output_details)

    def test_initialization_with_invalid_model(self):
        with self.assertRaises(RuntimeError):
            ConcreteTFLite(self.invalid_model_path)

    def test_inference(self):
        tflite_instance = ConcreteTFLite(self.valid_model_path)
        input_tensor = tf.constant([1.0, 2.0, 3.0])
        result = tflite_instance.inference(input_tensor)
        self.assertEqual(result.numpy().tolist(), input_tensor.numpy().tolist())
        self.assertIsInstance(result, tf.Tensor)


if __name__ == '__main__':
    unittest.main()
