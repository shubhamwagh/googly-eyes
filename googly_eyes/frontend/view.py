import os
import base64
import requests
import streamlit as st
from typing import Tuple, Optional


def display_title() -> None:
    """
    Displays the title of the Googlify Eyes App.
    """
    st.markdown("<h1 style='text-align:center;'>Googlify Eyes App</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='text-align:center;'>Upload an image, and click on 'Googlify' to make your face look funny!</h4>",
        unsafe_allow_html=True)


def get_backend_url() -> str:
    """
    Retrieves the backend URL from the environment variable or defaults to "http://localhost:8000/googlify".
    """
    url_backend = os.environ.get("URL_BACKEND", "http://localhost:8000/")
    return os.path.join(url_backend, "googlify")


def upload_image() -> Optional[st.file_uploader]:
    """
    Displays the file uploader for image selection.

    Returns:
    - Optional[st.file_uploader]: The uploaded image file.
    """
    return st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"])


def display_image_viewer(uploaded_file: st.file_uploader) -> Tuple[st.button, st.empty]:
    """
    Displays the image viewer and Googlify button.

    Args:
    - uploaded_file (st.file_uploader): The uploaded image file.

    Returns:
    - Tuple[st.button, st.empty]: The Googlify button and the Streamlit block for displaying images.
    """
    _, centre_col, _ = st.columns([3, 1, 3])
    googlify_button = centre_col.button("Googlify")

    image_viewer = st.empty()
    image_viewer.image(uploaded_file, use_column_width=True)

    return googlify_button, image_viewer


def googlify(googlify_button: st.button, uploaded_file: st.file_uploader, url_backend: str) -> Optional[bytes]:
    """
    Handles the Googlify process when the Googlify button is clicked.

    Args:
    - googlify_button (st.button): The Googlify button.
    - uploaded_file (st.file_uploader): The uploaded image file.
    - url_backend (str): The URL for the Googlify backend.

    Returns:
    - decoded_image (bytes): The decoded Googlify image.
    """
    if googlify_button:
        if hasattr(uploaded_file, 'name') and uploaded_file.name:
            filename = uploaded_file.name
        else:
            filename = "uploaded_image"

        files = {"image": (filename, uploaded_file, "image/*")}
        response = requests.post(url_backend, files=files)

        if response.status_code == 200:
            answer = response.json()
            decoded_image = base64.b64decode(answer["result"])
            return decoded_image
        else:
            st.error(f"{response.status_code}: Error processing the image!")

    return None


def display_footer() -> None:
    """
    Displays the footer.
    """
    st.text("Funny Faces Inc - All rights reserved ©")


def main():
    display_title()
    url_backend = get_backend_url()
    uploaded_file = upload_image()

    if uploaded_file is not None:
        googlify_button, image_viewer = display_image_viewer(uploaded_file)
        decoded_image = googlify(googlify_button, uploaded_file, url_backend)

        if decoded_image:
            image_viewer.image(decoded_image, use_column_width=True)

    display_footer()


if __name__ == "__main__":
    main()
