"""
    Запускать на бортовом компьютере беспилотника.

    Используем класс Arduino для управления скоростью и
    рулевыми колёсами беспилотного автомобиля.

    В конце пример ограничения дистанции, которую проедет беспилотник.
"""

from Arduino_A26 import Arduino
import time

# PORT = '/dev/ttyS0'
# PORT = '/dev/ttyUSB0'
PORT = 'COM4'

# пример обработки уже принятых сообщений от микроконтроллера
# интерпретирует текст сообщения и выполняет соответствующие действия
def parseMessage(msg):

    # простое сообщение из одного текстового поля, проверяем целиком
    if msg == "OK_STOP":
        print("\n***Response from Arduino***")
        print(f"Text: <{msg}>")
        return True

    # остались сообщения из двух полей, текст и значение
    parts = msg.split(':')
    if len(parts) <=1: # если часть одна, то ":" не оказалось
        return False  # сообщение не соответствует формату
    value = int(parts[1])  # переводим в число вторую часть строки от пользователя

    if parts[0] == "STAT_DIST":
        print("\n***Response from Arduino***")
        print(f"Text: <{parts[0]}>")
        print(f"Value: <{value}>")
        return True

    if parts[0] == "OK_DIST":
        print("\n***Response from Arduino***")
        print(f"Text: <{parts[0]}>")
        print(f"Value: <{value}>")
        return True

    return False  # сообщение не соответствует формату протокола


arduino = Arduino(PORT)  # создаём экземпляр класса

# мигаем светодиодом, сообщений от Arduino не ждём
led_state = False
for i in range(30):
    led_state = i % 2
    arduino.set_led(led_state)
    time.sleep(0.2)

# # выставляем разные значения скорости и угла поворота колёс
angle_list = [120, 90, 70, 120, 70, 90]
speed_list = [1630, 1310, 1500, 1330, 1630, 1650]
for i in range(6):
    arduino.set_angle(angle_list[i])
    arduino.set_speed(speed_list[i])
    time.sleep(1)

# останавливаем двигатель
# получаем от Ардуино подтверждение того, что остановка выполнена
arduino.stop()  # Ардуино ответит сообщением
start_time = time.monotonic()
while arduino.receive_message() is False: # крутим цикл, пока не получим сообщение
    if time.monotonic() - start_time > 1: # через секунду выходим из цикла
        print("!!! The stop command is missing !!!")
        arduino.close()  # закрываем соединение
        exit()

# Устанавливаем дистанцию, которую необходимо проехать
# Дистанция измеряется в срабатываниях энкодера
arduino.set_dist(2000)
# принимаем подтверждение от Arduino
start_time = time.monotonic()
while arduino.receive_message() is False: # крутим цикл, пока не получим сообщение
    if time.monotonic() - start_time > 1: # через секунду выходим из цикла
        print("!!! The DIST command is missing !!!")
        arduino.close()  # закрываем соединение
        exit()

# Беспилотник будет двигаться с заданной вами скоростью до тех пор,
# пока не отсчитает указанное число срабатываний энкодера
# Когда дистанция будет пройдена, беспилотник остановится

# Запрашивайте у Ардуино оставшееся значение дистанции, которое необходимо проехать,
# чтобы отследить факт остановки.
# После остановки беспилотник будет возвращать дистанцию 0

# Вы можете останавливать беспилотник и изменять его скорость,
# на отсчёт дистанции это не повлияет
# Энкодер одинаково срабатывает при движении вперёд и назад

# Задаём скорость для движения вперёд
arduino.set_speed(1630)

# В течение 10 секунд запрашиваем у автомобиля текущее значение дистанции
start_time = time.monotonic()
while (True):  # пример цикла обработки сообщений
    arduino.get_dist()
    while arduino.receive_message():  # если сообщение есть, то
        msg = arduino.get_msg_from_buffer()  # получаем его текст без "*" и "|"
        parseMessage(msg)  # обрабатываем сообщения

    time.sleep(0.5)  # 2 запроса в секунду

    if time.monotonic() - start_time > 10:
        break

arduino.close()  # закрываем соединение
print("\n Program complete.")
