import os
import unittest
from unittest.mock import MagicMock, patch, call, PropertyMock
from io import BytesIO
import base64

from googly_eyes.frontend.view import (
    display_title,
    get_backend_url,
    upload_image,
    display_image_viewer,
    googlify,
    display_footer,
    main,
)


class TestView(unittest.TestCase):
    def setUp(self):
        # Set up Streamlit mock for testing
        self.st_patch = patch("streamlit.empty")
        self.mock_st_empty = self.st_patch.start()

    def tearDown(self):
        # Clean up Streamlit mock
        self.st_patch.stop()

    def test_display_title(self):
        # Ensure display_title calls the correct Streamlit functions
        mock_st_markdown = MagicMock()
        with patch("streamlit.markdown", mock_st_markdown):
            display_title()

        call1 = call("<h1 style='text-align:center;'>Googlify Eyes App</h1>",
                     unsafe_allow_html=True)
        call2 = call(
            "<h4 style='text-align:center;'>Upload an image, and click on 'Googlify' to make your face look funny!</h4>",
            unsafe_allow_html=True)

        mock_st_markdown.assert_has_calls([call1, call2], any_order=False)

    def test_get_backend_url(self):
        # Ensure get_backend_url returns the correct URL
        os.environ["URL_BACKEND"] = "http://example.com/"
        self.assertEqual(get_backend_url(), "http://example.com/googlify")

    def test_upload_image(self):
        # Ensure upload_image returns a file uploader object
        with patch("streamlit.file_uploader", return_value="fake_uploaded_file"):
            self.assertEqual(upload_image(), "fake_uploaded_file")

    def test_display_image_viewer(self):
        # Test that display_image_viewer calls the correct Streamlit functions
        mock_st_columns = MagicMock()
        mock_st_column = MagicMock()
        mock_st_empty = MagicMock()
        mock_st_button = PropertyMock()

        # Patching streamlit.columns and streamlit.button
        with patch("streamlit.columns", mock_st_columns), \
                patch("streamlit.delta_generator.DeltaGenerator", mock_st_column), \
                patch("streamlit.delta_generator.DeltaGenerator.button", mock_st_button), \
                patch("streamlit.empty", mock_st_empty):
            # Mocking the st.columns([3, 1, 3]) call to return a tuple with three elements
            column_return_value = (MagicMock(), mock_st_column, MagicMock())
            mock_st_columns.return_value = column_return_value
            result = display_image_viewer("fake_uploaded_file")

        mock_st_columns.assert_called_with([3, 1, 3])
        mock_st_button.assert_called_once_with("Googlify")
        mock_st_empty.assert_called_once()
        mock_st_empty.return_value.image.assert_called_with("fake_uploaded_file", use_column_width=True)

        self.assertEqual(result, (mock_st_button.return_value, mock_st_empty.return_value))

    @patch("requests.post")
    @patch("streamlit.error")
    def test_googlify_success(self, mock_st_error, mock_requests_post):
        # Test googlify handles success correctly
        fake_image_data = b"fake_image_data"
        fake_base64_image = base64.b64encode(fake_image_data).decode('utf-8')

        mock_response = MagicMock()
        mock_response.status_code = 200

        # Use the base64 encoded image in the mock response
        mock_response.json.return_value = {"result": fake_base64_image}
        mock_requests_post.return_value = mock_response

        mock_file = BytesIO(fake_image_data)

        result = googlify(True, mock_file, "fake_url_backend")

        mock_requests_post.assert_called_with("fake_url_backend",
                                              files={"image": ("uploaded_image", mock_file, "image/*")})
        mock_st_error.assert_not_called()

        self.assertEqual(result, fake_image_data)

    @patch("requests.post")
    @patch("streamlit.error")
    def test_googlify_failure(self, mock_st_error, mock_requests_post):
        # Test googlify handles failure correctly
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_requests_post.return_value = mock_response
        mock_file = BytesIO(b"fake_image_data")

        result = googlify(True, mock_file, "fake_url_backend")

        mock_requests_post.assert_called_with("fake_url_backend",
                                              files={"image": ("uploaded_image", mock_file, "image/*")})
        mock_st_error.assert_called_with("500: Error processing the image!")

        self.assertIsNone(result)

    @patch("streamlit.text")
    def test_display_footer(self, mock_st_text):
        # Ensure display_footer calls the correct Streamlit function
        display_footer()
        mock_st_text.assert_called_with("Funny Faces Inc - All rights reserved ©")

    @patch("googly_eyes.frontend.view.display_title")
    @patch("googly_eyes.frontend.view.get_backend_url")
    @patch("googly_eyes.frontend.view.upload_image")
    @patch("googly_eyes.frontend.view.display_image_viewer")
    @patch("googly_eyes.frontend.view.googlify")
    @patch("googly_eyes.frontend.view.display_footer")
    def test_main(self, mock_display_footer, mock_googlify, mock_display_image_viewer, mock_upload_image,
                  mock_get_backend_url, mock_display_title):
        mock_st_empty = MagicMock()
        mock_st_empty.image = MagicMock()
        mock_get_backend_url.return_value = "fake_url_backend"
        mock_upload_image.return_value = "fake_uploaded_file"
        mock_display_image_viewer.return_value = ("fake_button", mock_st_empty)
        mock_googlify.return_value = BytesIO(b"fake_decoded_image")

        main()

        mock_display_title.assert_called_once()
        mock_get_backend_url.assert_called_once()
        mock_upload_image.assert_called_once()
        mock_display_image_viewer.assert_called_once_with("fake_uploaded_file")
        mock_googlify.assert_called_once_with("fake_button", "fake_uploaded_file", "fake_url_backend")
        mock_display_footer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
