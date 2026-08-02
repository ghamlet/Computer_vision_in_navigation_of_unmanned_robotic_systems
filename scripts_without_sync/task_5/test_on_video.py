#!/usr/bin/env python3
"""Тест модели YOLOv4-tiny (working / notworking) на видеозаписи.

Модель: model/yolov4-tiny-svetofor_best_weights.weights (OpenCV DNN)
По умолчанию: records/received_20260802_140719.avi

Примеры:
  python3 test_on_video.py
  python3 test_on_video.py --save result.avi
  python3 test_on_video.py --conf 0.25 --skip 1
  python3 test_on_video.py --input-size 320   # уменьшить разрешение для ускорения
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
RECORDS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'records'))

CFG = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor.cfg')
WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor_best_weights.weights')
NAMES = os.path.join(MODEL_DIR, 'svetofor.names')

DEFAULT_VIDEO = os.path.join(RECORDS_DIR, 'received_20260802_140719.avi')
DEFAULT_INPUT_SIZE = 416      # размер по умолчанию (можно уменьшить для скорости)

COLORS = {
    'notworking': (0, 0, 255),
    'working': (0, 255, 0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def check_model_files():
    missing = []
    for label, path in [('cfg', CFG), ('weights', WEIGHTS), ('names', NAMES)]:
        if not os.path.exists(path):
            missing.append(f'{label}: {path}')
    if missing:
        print('[ERROR] Model files not found:')
        for m in missing:
            print(f'  - {m}')
        sys.exit(1)

    print('[INFO] Model files OK:')
    for label, path in [('cfg', CFG), ('weights', WEIGHTS), ('names', NAMES)]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'  - {label}: {os.path.basename(path)} ({size_mb:.1f} MB)')


def load_detector(conf_threshold, nms_threshold):
    check_model_files()
    with open(NAMES) as f:
        class_names = [line.strip() for line in f if line.strip()]

    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    out_layers = net.getUnconnectedOutLayersNames()
    print(f'[INFO] Model loaded, classes={class_names}, outputs={out_layers}')
    return net, out_layers, class_names, conf_threshold, nms_threshold


def detect(net, out_layers, class_names, frame, conf_threshold, nms_threshold,
           input_size):
    h, w = frame.shape[:2]
    # Главное изменение: используем переданный input_size вместо константы
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_size, input_size),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_threshold:
                continue
            cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

    results = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        indices = np.array(indices).flatten() if len(indices) else []
        for i in indices:
            label = class_names[class_ids[i]]
            results.append({
                'class': label,
                'score': confidences[i],
                'box': boxes[i],
                'label': f'{label} {confidences[i] * 100:.0f}%',
            })
    return results


def draw_detections(frame, detections):
    for det in detections:
        x, y, bw, bh = det['box']
        color = COLORS.get(det['class'], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, det['label'], (x, max(y - 8, 16)),
                    FONT, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser(description='Test svetofor model on video')
    parser.add_argument('--source', '-s', default=DEFAULT_VIDEO,
                        help=f'Video path (default: {DEFAULT_VIDEO})')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip N frames between detections (0 = every frame)')
    parser.add_argument('--save', '-o', default=None, help='Save annotated video')
    parser.add_argument('--no-show', action='store_true', help='Do not open cv2 window')
    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help=f'Input size for the network (default: {DEFAULT_INPUT_SIZE}). '
                             'Decrease to speed up (e.g. 320, 288).')
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f'[ERROR] Video not found: {args.source}')
        return 1

    # Загружаем модель
    net, out_layers, class_names, conf_threshold, nms_threshold = load_detector(
        args.conf, args.nms)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {args.source}')
        return 1

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'[INFO] Video: {args.source}')
    print(f'[INFO] {width}x{height} @ {fps_in:.1f} fps, frames={total}')
    print(f'[INFO] Using input size: {args.input_size}x{args.input_size}')

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps_in, (width, height))
        print(f'[INFO] Saving to: {args.save}')

    frame_idx = 0
    det_frames = 0
    total_dets = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            detections = []
            if args.skip == 0 or (frame_idx - 1) % (args.skip + 1) == 0:
                detections = detect(net, out_layers, class_names, frame,
                                    conf_threshold, nms_threshold,
                                    args.input_size)   # передаём размер
                if detections:
                    det_frames += 1
                    total_dets += len(detections)
                    print(f'[frame {frame_idx}/{total}] {len(detections)} det(s):')
                    for det in detections:
                        print(f'  - {det["class"]}: {det["score"] * 100:.1f}% '
                              f'box={det["box"]}')

            draw_detections(frame, detections)
            cv2.putText(frame, f'frame {frame_idx}/{total}', (10, 24),
                        FONT, 0.6, (255, 255, 255), 2)

            if writer:
                writer.write(frame)
            if not args.no_show:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == 32:
                    while True:
                        k = cv2.waitKey(100) & 0xFF
                        if k in (27, ord('q'), 32):
                            break

            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f'[progress] {frame_idx}/{total} frames, '
                      f'{frame_idx / elapsed:.1f} fps processing')

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f'\n[INFO] Done: {frame_idx} frames in {elapsed:.1f}s')
    print(f'[INFO] Frames with detections: {det_frames}/{frame_idx}')
    print(f'[INFO] Total detections: {total_dets}')
    return 0


if __name__ == '__main__':
    sys.exit(main())