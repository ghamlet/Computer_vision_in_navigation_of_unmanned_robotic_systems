#!/usr/bin/env python3
"""Имитатор квадрокоптера: слушает команду «на взлёт» по UDP и выводит её.

Запускается на компьютере оператора. Адрес/порт должны совпадать
с QUAD_HOST/QUAD_PORT в scripts/task_1/StadionRunner.py.
"""

import socket

HOST = '127.0.0.1'
PORT = 9000


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print(f'[QUAD] Listening on udp://{HOST}:{PORT}...')

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            msg = data.decode('utf-8', errors='replace')
            print(f'[QUAD] Received from {addr}: {msg}')
            if msg == 'на взлёт':
                print('[QUAD] ВЗЛЁТ!')
    except KeyboardInterrupt:
        print('\n[QUAD] Stopped')
    finally:
        sock.close()


if __name__ == '__main__':
    main()
