#!/usr/bin/env python3
"""
MJPEG stream receiver + lane detection (like StadionRunner.py).
Connects to video stream on 172.17.49.10:8081, processes frames, shows binarized result.
"""

import sys
import threading
import time
import urllib.request
import cv2
import numpy as np

from road_utils import *

# ========== CONFIG ==========
STREAM_URL = 'http://172.17.49.10:8081/'  # video_sender.py on robot

# Lane detection config (from original StadionRunner.py)
THRESHOLD = 200

latest_frame = None
frame_lock = threading.Lock()
running = True


def fetch_mjpeg_stream():
    """Background thread: fetches MJPEG stream and extracts frames."""
    global latest_frame, running

    print(f'[INFO] Connecting to MJPEG stream: {STREAM_URL}')
    req = urllib.request.urlopen(STREAM_URL, timeout=10)

    JPEG_SOI = b'\xff\xd8'
    JPEG_EOI = b'\xff\xd9'
    buf = b''
    frame_count = 0

    try:
        while running:
            chunk = req.read(65536)
            if not chunk:
                print('[DEBUG] No chunk received, stream ended')
                break
            buf += chunk
            while True:
                start = buf.find(JPEG_SOI)
                end = buf.find(JPEG_EOI)
                if start == -1 or end == -1 or end < start:
                    break
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f'[DEBUG] Fetched {frame_count} frames, latest: {frame.shape}')
    except Exception as e:
        print(f'[ERROR] Stream fetch error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        req.close()
        print('[DEBUG] Fetch thread ended')


def main():
    global running, latest_frame

    # Start MJPEG fetch thread
    fetch_thread = threading.Thread(target=fetch_mjpeg_stream, daemon=True)
    fetch_thread.start()
    print('[DEBUG] Fetch thread started, entering wait loop...')

    # Wait for first frame
    print('[INFO] Waiting for first frame...')
    wait_count = 0
    while True:
        with frame_lock:
            got_frame = latest_frame is not None
        if got_frame or not running:
            break
        time.sleep(0.1)
        wait_count += 1
        if wait_count % 10 == 0:
            with frame_lock:
                lf = latest_frame
            print(f'[DEBUG] Main still waiting... latest_frame is None: {lf is None}, running: {running}')

    if latest_frame is None:
        print('[ERROR] Failed to get frame from stream')
        return 1

    print('[INFO] First frame received, starting processing loop')

    find_lines = centre_mass2
    last_err = 0

    try:
        while True:
            start_time = time.time()

            # Get latest frame from stream
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # --- Lane detection (exactly like original StadionRunner.py) ---
            frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
            orig_frame = frame.copy()
            frame = cv2.resize(frame, SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bin = cv2.inRange(gray, THRESHOLD, 255)

            cv2.imshow("bin", bin)

            wrapped = trans_perspective(bin, TRAP, RECT, SIZE)
            cv2.imshow("wrapped", wrapped)
            left, right = find_lines(wrapped)

            # ПИД-регулятор
            err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
            err = -err
            angle = int(90 + KP * err + KD * (err - last_err))
            last_err = err

            angle = min(max(45, angle), 135)

            print(f'angle: {angle}, left: {left}, right: {right}')

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            if fps < 10:
                print(f'[WARNING] FPS too low: {fps:.1f}')

            cv2.waitKey(1)

    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        running = False
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())