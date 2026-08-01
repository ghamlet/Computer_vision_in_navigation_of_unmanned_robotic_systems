#!/usr/bin/env python3
"""Видео анализ: только закрашивание зон (верх и низ кадра)."""

import os
import sys
import argparse
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TOP_FILL_H = 100
BOTTOM_FILL_Y = 640
FILL_X = 870
FILL_COLOR = (0, 255, 0)


def fill_zones(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, min(TOP_FILL_H, h)), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, min(BOTTOM_FILL_Y, h)), (w, h), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, 0), (min(FILL_X, w), h), FILL_COLOR, -1)


def resolve_source(source):
    if source != 'auto':
        return source
    video = next((f for f in os.listdir(SCRIPT_DIR)
                  if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))), None)
    return os.path.join(SCRIPT_DIR, video) if video else 0


def main():
    parser = argparse.ArgumentParser(description='Video with zone filling')
    parser.add_argument('--source', '-s', default='auto',
                        help='Video source (file, camera index, URL); default: video from this dir')
    args = parser.parse_args()

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {source}')
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] Resolution: {width}x{height}')
    print(f'[INFO] Left fill x: 0-{FILL_X}px')
    print(f'[INFO] Top fill: {TOP_FILL_H}px, Bottom fill from: {BOTTOM_FILL_Y}px')

    paused = False
    try:
        while True:
            if paused:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key in (ord('f'), ord('F')):
                    paused = False
                continue

            ret, frame = cap.read()
            if not ret:
                break

            fill_zones(frame)

            cv2.imshow('Video (F - pause, ESC/q - stop)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in (ord('f'), ord('F')):
                paused = True
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())