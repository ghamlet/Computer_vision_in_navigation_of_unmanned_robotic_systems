""" Детекция дорожных знаков.
    Можно использовать как отдельный модуль в других скриптах.
"""

import sys
from pathlib import Path

import cv2
import yolopy

MODEL_DIRS = [Path.cwd(), Path('/home/avt_user/Base_Code'), Path('/home/avt_user/PROGRAMMS')]


def find_model_file(name):
    for d in MODEL_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


class SignDetector:
    def __init__(self, detect_class=6, missed_frames_limit=5):
        self.detect_class = detect_class
        self.missed_frames_limit = missed_frames_limit
        self.sign_seen = False
        self.missed_frames = 0

        classes_file = find_model_file('classes.txt')
        model_file = find_model_file('yolo_uint8.tmfile')
        if classes_file is None or model_file is None:
            raise FileNotFoundError('classes.txt and yolo_uint8.tmfile not found')

        with open(classes_file) as file:
            self.class_names = file.read().splitlines()

        self.model = yolopy.Model(str(model_file), use_uint8=True, use_timvx=True, cls_num=10)
        self.model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])

    def process_frame(self, frame):
        """Обрабатывает кадр, возвращает (should_stop, labeled_frame, found_sign)"""
        classes, scores, boxes = self.model.detect(frame)

        labeled_frame = frame.copy()
        found = False
        for classid, score, box in zip(classes, scores, boxes):
            if classid != self.detect_class:
                continue
            found = True
            label = f'{self.class_names[classid]} [{score * 100:.2f}%]'
            cv2.rectangle(labeled_frame, box, (0, 255, 0), 2)
            cv2.putText(labeled_frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        should_stop = False
        if found:
            self.sign_seen = True
            self.missed_frames = 0
            print(f'[SIGN] {self.class_names[self.detect_class]} detected')
        else:
            if self.sign_seen:
                self.missed_frames += 1
                print(f'[SIGN] not found: {self.missed_frames}/{self.missed_frames_limit}')
                if self.missed_frames >= self.missed_frames_limit:
                    print('[STOP] СТОП')
                    should_stop = True
                    self.sign_seen = False
                    self.missed_frames = 0

        return should_stop, labeled_frame, found


# HTTP streaming functionality (original) - runs only when executed directly
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    CAMERA_ID = '/dev/video0'
    HOST = '0.0.0.0'
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    JPEG_QUALITY = 70

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print('[ERROR] Cannot open camera ID:', CAMERA_ID)
        sys.exit(1)

    jpeg_lock = threading.Lock()
    current_jpeg = None
    detector = SignDetector()

    def grab_loop():
        global current_jpeg
        for i in range(30):
            cap.read()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            should_stop, labeled_frame, _ = detector.process_frame(frame)

            ok, buf = cv2.imencode('.jpg', labeled_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    current_jpeg = buf.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with jpeg_lock:
                    frame = current_jpeg
                if frame is None:
                    continue
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break

        def log_message(self, *args):
            pass

    t = threading.Thread(target=grab_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f'[INFO] Streaming sign detections on http://{HOST}:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        server.server_close()
        cap.release()


if __name__ == '__main__':
    main()
