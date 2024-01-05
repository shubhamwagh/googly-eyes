import os
import unittest
from fastapi.testclient import TestClient
from googly_eyes.backend.server import app


class TestServerEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.sample_img = os.path.join(cwd, "../../assets/misc/multi_face.png")
        self.random_file = os.path.join(cwd, "../../assets/misc/random_file.txt")

    def test_welcome_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"msg": "Welcome to Funny Faces Inc - Company Main Page"})

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

    def test_valid_image_googlify_endpoint(self):
        with open(self.sample_img, 'rb') as image_file:
            files = {'image': ('image.png', image_file, 'image/png')}
            response = self.client.post("/googlify", files=files)

        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.json())

    def test_non_image_file_googlify_endpoint(self):
        with open(self.random_file, 'rb') as text_file:
            files = {'image': ('text_file.txt', text_file, 'text/plain')}
            response = self.client.post("/googlify", files=files)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {
            "detail": [{
                "loc": ["file"],
                "msg": "Uploaded file must be an image file",
                "type": "validation_error"
            }]
        })


if __name__ == '__main__':
    unittest.main()