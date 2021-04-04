import sys
import time
import numpy as np
import cv2
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

from utils.visualisation import show_detection_results


# CONSTANTS
HEIGHT, WIDTH = (640, 960)
N_SKIP = 1


def main():
    # Load YOLOv4
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=100,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.5,
    )
    model.load_weights("../model/yolov4.h5")
    model.summary()

    # Run on video
    cap = cv2.VideoCapture('/Users/stin/Documents/data/traffic/view1.mp4')
    frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, img_raw = cap.read()

        if not ret or (frame > total_frames):
            # TODO: write counts here
            frame = 0
            break

        frame += N_SKIP

        # Image
        img = img_raw.copy()
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0

        # Inference
        start_time = time.time()
        boxes, scores, classes, valid_detections = model.predict(img)
        duration = time.time() - start_time
        print(f"Inference elapsed time: {duration} s")

        # Visualise
        show_detection_results(
            img[0],
            boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
            scores[0],
            classes[0].astype(int),
        )

        sys.exit()


if __name__ == '__main__':
    main()
