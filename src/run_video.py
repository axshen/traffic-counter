# https://pypi.org/project/tf2-yolov4/

import numpy as np
import cv2
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt

# RAW IMAGE SHAPE
HEIGHT, WIDTH = (640, 960)

# SKIP
N_SKIP = 1

# COCO classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, boxes, scores, classes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl % 6], linewidth=3))
          text = f'{CLASSES[cl]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def read_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    img = tf.expand_dims(img, axis=0) / 255.0
    return img


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
        img = np.expand_dims(img, axis=0)

        # Inference
        boxes, scores, classes, valid_detections = model.predict(img)

        # Visualise
        plot_results(
            img[0],
            boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
            scores[0],
            classes[0].astype(int),
        )


if __name__ == '__main__':
    main()
