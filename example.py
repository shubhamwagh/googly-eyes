import os
import sys
import cv2
import argparse
from googly_eyes.backend.lib.GooglyEyes import GooglifyEyes
from googly_eyes.backend.lib.utils import io_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Googlify Eyes")
    parser.add_argument("image_path", type=str, nargs="?", default="assets/misc/multi_face.png",
                        help="Path to the input image file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: The file '{args.image_path}' does not exist.")
        sys.exit(1)

    params = io_utils.load_config()

    googlify_eyes = GooglifyEyes(params)

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    out = googlify_eyes.generate(img)
    cv2.imshow("Googlify Eyes", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
