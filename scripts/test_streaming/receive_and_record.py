#!/usr/bin/env python3
"""Receive MJPEG stream from robot and save as video file. Run on PC."""

import sys
import os
import urllib.request
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print('[ERROR] OpenCV required on PC for video recording')
    sys.exit(1)

DEFAULT_URL = 'http://172.17.49.10:8081'
DEFAULT_OUTPUT_DIR = Path('/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/records')
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
# If output file not specified, create timestamped filename in records dir
if len(sys.argv) > 2:
    OUTPUT_FILE = Path(sys.argv[2])
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_FILE = DEFAULT_OUTPUT_DIR / f'received_{timestamp}.avi'

JPEG_SOI = b'\xff\xd8'
JPEG_EOI = b'\xff\xd9'


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Connecting to {URL} ...')
    req = urllib.request.urlopen(URL, timeout=10)
    print(f'[INFO] Connected. Recording to {OUTPUT_FILE}')

    writer = None
    frame_size = None
    buf = b''
    frame_count = 0

    try:
        while True:
            chunk = req.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(JPEG_SOI)
                end = buf.find(JPEG_EOI)
                if start == -1 or end == -1 or end < start:
                    break
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]

                frame_count += 1
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                if writer is None:
                    h, w = img.shape[:2]
                    frame_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, 30.0, frame_size)
                    if not writer.isOpened():
                        print('[ERROR] Cannot open video writer')
                        return
                    print(f'[INFO] Video writer opened: {frame_size[0]}x{frame_size[1]} @ 30fps (MJPG/AVI)')

                writer.write(img)

                cv2.imshow('Robot Stream (ESC/q to quit)', img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print(f'\n[INFO] Stopped by user. Saved {frame_count} frames.')
                    return

                if frame_count % 30 == 0:
                    print(f'[INFO] {frame_count} frames recorded...')

    except KeyboardInterrupt:
        print(f'\n[INFO] Interrupted. Saved {frame_count} frames to {OUTPUT_FILE}')
    finally:
        req.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f'[INFO] Done. Video saved: {str(OUTPUT_FILE)}')


if __name__ == '__main__':
    main()