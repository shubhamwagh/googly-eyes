import tensorflow as tf
from googly_eyes.backend.lib.utils import image_utils
from abc import ABC, abstractmethod

from typing import Any


class TFLite(ABC):
    """
    Abstract base class for TensorFlow Lite (TFLite) models.

    Attributes:
        image_info (image_utils.ModelInputInfo): Information about the model input image.
        interpreter (tf.lite.Interpreter): TensorFlow Lite Interpreter variable.
        model_input_details (List[dict]): Details of the model input.
        model_input_info (image_utils.ModelInputInfo): Information about the model input shape.
        model_output_details (List[dict]): Details of the model output.
        model_path (str): Path to the TFLite model file.

    Methods:
        initialise_model(): Initializes the TFLite model.
        get_model_input_details(): Retrieves details of the model input.
        get_model_output_details(): Retrieves details of the model output.
        inference(input_tensor: tf.Tensor) -> Any: Abstract method for performing inference.
    """

    def __init__(self, model_path: str):
        """
        Initializes the TFLite model.

        Args:
            model_path (str): Path to the TFLite model file.
        """
        self.image_info = None
        self.interpreter = None  # tf lite Interpreter variable

        self.model_input_details = None  # model input details
        self.model_input_info = None
        self.model_output_details = None  # model output details
        self.model_path = model_path
        self.initialise_model()

    def initialise_model(self) -> None:
        """
        Initializes the TFLite model.
        Raises:
            RuntimeError: If there is an issue with loading the TFLite model.
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f'TFlite model error: {e}')
        # Get model info
        self.get_model_input_details()
        self.get_model_output_details()

    def get_model_input_details(self) -> None:
        """
          Retrieves details of the model input.
        """
        self.model_input_details = self.interpreter.get_input_details()
        input_shape = self.model_input_details[0]['shape']
        self.model_input_info = image_utils.ModelInputInfo(height=input_shape[1], width=input_shape[2],
                                                           channels=input_shape[3])

    def get_model_output_details(self) -> None:
        """
          Retrieves details of the model output.
        """
        self.model_output_details = self.interpreter.get_output_details()

    @abstractmethod
    def inference(self, input_tensor: tf.Tensor) -> Any:
        """
        Abstract method for performing inference.
        Args:
            input_tensor (tf.Tensor): Input tensor for inference.
        Returns:
            Any: Inference result.
        """
        raise NotImplementedError('Implement inference method for your model')
