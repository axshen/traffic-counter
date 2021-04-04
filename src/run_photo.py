import time
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from utils.visualisation import show_detection_results


# Constants
HEIGHT, WIDTH = (640, 960)


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

    # Load image
    img_file = '/Users/stin/Documents/data/Crowd_PETS09/S2/L2/Time_14-55/View_001/frame_0000.jpg'  # noqa
    img = read_image(img_file)

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


if __name__ == '__main__':
    main()
