# Use YOLOv4 for inference
# https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-about/

import time
import cv2
from yolov4.tf import YOLOv4


def main():
    yolo = YOLOv4()
    yolo.config.parse_names("model/coco.names")
    yolo.config.parse_cfg("model/yolov4.cfg")

    yolo.make_model()
    yolo.load_weights("model/yolov4.weights", weights_type="yolo")
    yolo.summary(summary_type="yolo")
    yolo.summary()

    # inference on first frame


if __name__ == '__main__':
    main()
