import threading
import os
import cv2
from flask import Flask, request

# Создаём Flask-приложение для приёма POST-запросов.
app = Flask(__name__)

# Путь к исходной картинке, которая будет использоваться как база для отрисовки.
MAP_IMAGE_PATH = "image.jpg"

# Координаты станций на карте.
# Каждая пара — это (x, y) центр круга станции.
STATION_COORDS = [(200, 300), (400, 300)]

# Радиус круга, которым отображается станция.
STATION_RADIUS = 12

# Половина стороны квадрата, которым отображается мусор.
DEBRIS_HALF = 9

# Порт, на котором запускается сервер.
PORT = 5001

# Счётчик полученных сообщений о станциях.
# Используется, чтобы понять, к какой станции относится очередной статус.
stations_counter = 0

# Центр станции, которая сейчас находится в процессе зарядки.
# Нужен для отрисовки стадий зарядки 1–5.
station_charge_center = None

# Список координат всего мусора, который был добавлен на карту.
trash_coords = []

# Флаг, показывающий, нужно ли обновить окно карты.
refresh_map_flag = True

# Блокировка для безопасной работы с общими данными из разных потоков.
lock = threading.Lock()

# Загружаем исходное изображение карты.
# original_map хранит чистую версию, а working_map — текущую изменяемую.
original_map = cv2.imread(MAP_IMAGE_PATH)
working_map = original_map.copy()

# Изначально рисуем все станции жёлтым цветом.
for cx, cy in STATION_COORDS:
    cv2.circle(working_map, (cx, cy), STATION_RADIUS, (0, 255, 255), -1)


@app.route("/eyecar", methods=["POST"])
def eyecar():
    global stations_counter, station_charge_center
    global refresh_map_flag, working_map

    # Получаем тело POST-запроса как строку.
    data = request.get_data(as_text=True)

    # Блокируем доступ к общим данным, чтобы потоки не мешали друг другу.
    with lock:
        # Если пришёл статус станции:
        # "Good" — станция исправна,
        # "Broken" — станция неисправна.
        if data in ("Good", "Broken"):
            stations_counter += 1
            idx = stations_counter - 1

            # Берём координаты очередной станции из списка.
            cx, cy = STATION_COORDS[idx]

            if data == "Good":
                # Исправная станция становится зелёной.
                color = (0, 255, 0)
                # Сохраняем центр этой станции как активную для дальнейших стадий зарядки.
                station_charge_center = (cx, cy)
            else:
                # Неисправная станция становится красной.
                color = (0, 0, 255)

            # Закрашиваем круг станции и затем рисуем обводку.
            cv2.circle(working_map, (cx, cy), STATION_RADIUS, color, -1)
            cv2.circle(working_map, (cx, cy), STATION_RADIUS, color, 2)

        # Если пришла стадия зарядки от "1" до "5".
        elif data in ("1", "2", "3", "4", "5"):
            stage = int(data)

            # Стадия рисуется только если известен центр активной станции.
            if station_charge_center is not None:
                cx, cy = station_charge_center

                # Сначала снова закрашиваем станцию зелёным.
                cv2.circle(working_map, (cx, cy), STATION_RADIUS, (0, 255, 0), -1)

                # По мере роста стадии добавляем чёрные четверти круга,
                # чтобы визуально показывать прогресс зарядки.
                if stage >= 2:
                    cv2.ellipse(working_map, (cx, cy), (STATION_RADIUS, STATION_RADIUS), 0, 0, 90, (0, 0, 0), -1)
                if stage >= 3:
                    cv2.ellipse(working_map, (cx, cy), (STATION_RADIUS, STATION_RADIUS), 0, 90, 180, (0, 0, 0), -1)
                if stage >= 4:
                    cv2.ellipse(working_map, (cx, cy), (STATION_RADIUS, STATION_RADIUS), 0, 180, 270, (0, 0, 0), -1)
                if stage >= 5:
                    cv2.ellipse(working_map, (cx, cy), (STATION_RADIUS, STATION_RADIUS), 0, 270, 360, (0, 0, 0), -1)

        # После любых изменений нужно обновить окно карты.
        refresh_map_flag = True

    return "", 200


@app.route("/fly", methods=["POST"])
def fly():
    global refresh_map_flag, working_map

    # Получаем тело POST-запроса как строку.
    data = request.get_data(as_text=True)

    # Блокируем доступ к общим данным, чтобы изменения были безопасными.
    with lock:
        # Если пришла команда сброса мусора:
        # удаляем все ранее нарисованные квадраты с карты,
        # восстанавливая соответствующие участки из original_map.
        if data == "reset":
            for x, y in trash_coords:
                x1, y1 = x - DEBRIS_HALF, y - DEBRIS_HALF
                x2, y2 = x + DEBRIS_HALF + 1, y + DEBRIS_HALF + 1
                working_map[y1:y2, x1:x2] = original_map[y1:y2, x1:x2].copy()

            # Очищаем список мусора.
            trash_coords.clear()

        else:
            # Иначе ожидается строка вида "x,y".
            x, y = map(int, data.split(","))

            # Сохраняем координаты мусора, чтобы потом можно было его удалить.
            trash_coords.append((x, y))

            # Рисуем мусор в виде оранжевого квадрата.
            cv2.rectangle(
                working_map,
                (x - DEBRIS_HALF, y - DEBRIS_HALF),
                (x + DEBRIS_HALF, y + DEBRIS_HALF),
                (0, 165, 255),
                -1
            )

        # После любых изменений нужно обновить окно карты.
        refresh_map_flag = True

    return "", 200


def display_loop():
    # Этот поток отвечает только за отображение окна с картой.
    global refresh_map_flag

    # Создаём окно OpenCV.
    cv2.namedWindow("Map")

    while True:
        # Если карта была изменена, обновляем изображение в окне.
        if refresh_map_flag:
            cv2.imshow("Map", working_map)
            refresh_map_flag = False

        # Ждём нажатие клавиш.
        # Если нажали Esc, выходим из цикла.
        if cv2.waitKey(100) & 0xFF == 27:
            break

    # Закрываем все окна OpenCV.
    cv2.destroyAllWindows()

    # Завершаем процесс полностью.
    os._exit(0)


# Запускаем отдельный поток для отображения карты.
# Он нужен, чтобы окно OpenCV работало параллельно с Flask-сервером.
display = threading.Thread(target=display_loop, daemon=True)
display.start()

# Запускаем Flask-сервер:
# - host="0.0.0.0" делает его доступным извне,
# - port=PORT задаёт порт,
# - debug=False отключает отладочный режим,
# - use_reloader=False предотвращает запуск второго процесса.
app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
