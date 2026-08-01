#!/usr/bin/env python3
"""Video recording script. Records video from camera or stream to a dedicated folder."""

import os
import sys
import cv2
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Record video to a dedicated folder')
    parser.add_argument('--source', '-s', default=0, help='Video source (camera index, video file, or RTSP/MJPEG URL)')
    parser.add_argument('--outdir', '-o', default='/home/avt_user/records', help='Output directory')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {args.source}')
        return 1

    if args.resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution[1])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'video_{timestamp}.{args.codec if args.codec != "mp4v" else "mp4"}'
    filepath = os.path.join(outdir, filename)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f'[ERROR] Cannot create video writer for: {filepath}')
        cap.release()
        return 1

    print(f'[INFO] Recording to: {filepath}')
    print(f'[INFO] Resolution: {width}x{height}, FPS: {fps}, Codec: {args.codec}')
    print('[INFO] Press ESC or q to stop recording')

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('[INFO] End of stream')
                break

            writer.write(frame)
            frame_count += 1

            cv2.imshow('Recording (ESC/q to stop)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print('[INFO] Stopped by user')
                break

            if frame_count % 30 == 0:
                print(f'[INFO] Recorded {frame_count} frames...')
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f'[INFO] Saved {frame_count} frames to {filepath}')

    return 0


if __name__ == '__main__':
    sys.exit(main())