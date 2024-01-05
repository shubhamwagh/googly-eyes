import cv2
import numpy as np
import mimetypes
import base64
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, Any

from googly_eyes.backend.lib.GooglyEyes import GooglifyEyes
from googly_eyes.backend.lib.utils import io_utils

# Create FastAPI instance
app = FastAPI()
params = {'path': 'assets/filters/googly_eye.png',
          'size_multiplier': 2.0,
          'size_percent_inc': 0.4,
          'centre_offset_percent': 0.3}

params = io_utils.load_config()
googlify_eyes = GooglifyEyes(params=params)


def is_image(filename: str) -> bool:
    """
    Check if the given filename corresponds to an image file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the filename corresponds to an image file, False otherwise.

    """
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type and mime_type.startswith('image')


@app.get("/")
def welcome() -> Dict[str, str]:
    """
    Endpoint to get a welcome message.

    Returns:
        dict: A welcome message.

    """
    return {"msg": "Welcome to Funny Faces Inc - Company Main Page"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint to check the health status.

    Returns:
        dict: The health status.

    """
    return {"status": "OK"}


@app.post("/googlify")
def googlify(image: UploadFile) -> Dict[str, Any]:
    """
    Endpoint to apply googly eyes filter to an uploaded image.

    Args:
        image (UploadFile): The image file to process.

    Raises:
        HTTPException: If the uploaded file is not an image file.

    Returns:
        dict: The result containing the encoded image.

    """
    if not is_image(image.filename):
        error_detail = [{
            "loc": ["file"],
            "msg": "Uploaded file must be an image file",
            "type": "validation_error"
        }]
        raise HTTPException(status_code=422, detail=error_detail)

    contents = image.file.read()
    nparr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    out = googlify_eyes.generate(img)
    _, encoded_img = cv2.imencode('.PNG', out)
    encoded_img = base64.b64encode(encoded_img)
    return {"result": encoded_img}
