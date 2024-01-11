import os
from locust import FastHttpUser, task, between


class GooglifyUser(FastHttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/misc/girl.jpg")

    @task
    def googlify_image(self):
        ext = os.path.splitext(self.img_path)[-1]
        with open(self.img_path, 'rb') as image_file:
            files = {'image': (f'image.{ext[1:]}', image_file, f'image/{ext[1:]}')}
            self.client.post("/googlify", files=files)
