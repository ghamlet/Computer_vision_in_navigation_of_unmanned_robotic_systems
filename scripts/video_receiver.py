#!/usr/bin/env python3
"""MJPEG stream receiver. Run on the PC.

Downloads frames from the robot's video_sender.py and saves them as JPEG
files. If OpenCV is available on the PC, frames are also shown in a window.

Usage on PC:
    python3 video_receiver.py [url] [outdir]
Defaults:
    url    = http://172.17.49.10:8080
    outdir = received_frames
Keys while the window is open:  ESC / q  - quit
"""

import os
import sys
import urllib.request

DEFAULT_URL = 'http://172.17.49.10:8080'
DEFAULT_OUTDIR = 'received_frames'

URL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTDIR

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

JPEG_SOI = b'\xff\xd8'
JPEG_EOI = b'\xff\xd9'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if HAS_CV2:
        print('[INFO] OpenCV available, frames will be displayed too')
    else:
        print('[INFO] OpenCV not found on this PC, frames are only saved to disk')

    req = urllib.request.urlopen(URL, timeout=10)
    print(f'[INFO] Connected to {URL}, receiving frames into {OUTDIR}/ ...')

    buf = b''
    index = 0
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

                index += 1
                path = os.path.join(OUTDIR, f'frame_{index:06d}.jpg')
                with open(path, 'wb') as f:
                    f.write(jpg)

                if HAS_CV2:
                    img = cv2.imdecode(np_frombuffer(jpg), cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow('Robot stream', img)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            print(f'[INFO] Stopped. Saved {index} frames to {OUTDIR}/')
                            return

                if index % 30 == 0:
                    print(f'[INFO] {index} frames received, latest: {path}')
    except KeyboardInterrupt:
        print(f'\n[INFO] Interrupted. Saved {index} frames to {OUTDIR}/')
    finally:
        req.close()
        if HAS_CV2:
            cv2.destroyAllWindows()


def np_frombuffer(data):
    import numpy as np
    return np.frombuffer(data, dtype=np.uint8)


if __name__ == '__main__':
    main()
