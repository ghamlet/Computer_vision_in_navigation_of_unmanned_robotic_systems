#!/usr/bin/env python3
"""Детекция светофоров YOLOv4-tiny через OpenCV (DNN).

Тест модели model/yolov4-tiny-svetofor_best_weights.weights на видео
из ../records или другом источнике.

Классы модели (model/svetofor.names):
  notworking - неисправный / погашенный светофор
  working    - рабочий светофор

Управление в окне:
  space - пауза,  ESC/q - выход
"""

import argparse
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
RECORDS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'records'))

CFG = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor.cfg')
WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor_best_weights.weights')
NAMES = os.path.join(MODEL_DIR, 'svetofor.names')

INPUT_SIZE = 416
COLORS = {
    'notworking': (0, 0, 255),
    'working': (0, 255, 0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_model(cfg=CFG, weights=WEIGHTS):
    if not os.path.exists(cfg):
        print(f'[ERROR] cfg not found: {cfg}')
        sys.exit(1)
    if not os.path.exists(weights):
        print(f'[ERROR] weights not found: {weights}')
        sys.exit(1)
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    return net, net.getUnconnectedOutLayersNames()


def postprocess(frame, outs, class_names, conf_threshold, nms_threshold):
    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_threshold:
                continue
            cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    indices = np.array(indices).flatten() if len(indices) else []

    detections = []
    for i in indices:
        detections.append((class_ids[i], confidences[i], boxes[i]))

    for class_id, confidence, box in detections:
        x, y, bw, bh = box
        label = class_names[class_id]
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        text = f'{label} {confidence * 100:.0f}%'
        cv2.putText(frame, text, (x, y - 8), FONT, 0.6, color, 2)
        print(f'  - {label}: {confidence * 100:.1f}% box={box}')

    return detections


def resolve_source(source):
    if source != 'auto':
        return source
    videos = sorted(
        f for f in os.listdir(RECORDS_DIR)
        if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))
    ) if os.path.isdir(RECORDS_DIR) else []
    if not videos:
        print(f'[ERROR] No videos found in {RECORDS_DIR}, pass --source explicitly')
        sys.exit(1)
    return os.path.join(RECORDS_DIR, videos[0])


def main():
    parser = argparse.ArgumentParser(description='YOLOv4-tiny svetofor detection via OpenCV')
    parser.add_argument('--source', '-s', default='auto',
                        help='Video source (file path, camera index, URL); '
                             'default: first video from ../records')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--skip', type=int, default=0, help='Skip N frames between detections')
    parser.add_argument('--save', '-o', default=None, help='Save annotated video to this file')
    args = parser.parse_args()

    net, out_layers = load_model()
    with open(NAMES) as f:
        class_names = f.read().splitlines()
    print(f'[INFO] Model loaded: {os.path.basename(CFG)}, classes={class_names}')

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {source}')
        return 1

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] Source: {source}, {width}x{height} @ {fps_in:.1f} fps')

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps_in, (width, height))

    paused = False
    frame_idx = 0
    try:
        while True:
            if paused:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key in (32, ord('p'), ord('P')):
                    paused = False
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.skip and (frame_idx - 1) % (args.skip + 1) != 0:
                pass
            else:
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                             swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(out_layers)
                n = len(postprocess(frame, outs, class_names, args.conf, args.nms))
                if n:
                    print(f'[frame {frame_idx}] {n} object(s):')

            cv2.putText(frame, f'frame {frame_idx}', (10, 20), FONT, 0.5,
                        (255, 255, 255), 1)
            if writer:
                writer.write(frame)

            cv2.imshow('Svetofor detection (space - pause, ESC/q - exit)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in (32, ord('p'), ord('P')):
                paused = True
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print('[INFO] Done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
