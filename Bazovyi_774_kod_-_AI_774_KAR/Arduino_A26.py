"""
    Класс Arduino для приёма и передачи сообщений от одноплатника к микроконтроллеру

    Сообщения, которые отправляет одноплатный компьютер:
    "*SET_SPEED:число|"  - задаём скорость движения беспилотника
    от 600 до 2400
    1500 - нулевая скорость. Чем меньше 1500, тем быстрее движение назад
    Чем больше 1500, тем быстрее движение вперёд

    "*SET_ANGLE:число|"  - задаём угол поворота рулевых колёс
    от 55 до 125
    90 - центральное положение

    "*SET_LED:число|"    - задаём состояние светодиода
    0 - выключить, 1 - включить

    "*STOP|"             - подаём двигателю сигнал остановки
    "*SET_DIST:число|"   - задаём число срабатываний энкодера, которое необходимо проехать
    отрицательное и 0 - выключают обратный отсчёт, положительное - включает отсчёт

    "*GET_DIST|"         - запрашиваем число срабатываний энкодера, которое осталось проехать

    Сообщения, которые отправляет микроконтроллер:
    "*OK_DIST:число|"    - подтверждаем получение команды и верного значения
    "*STAT_DIST:число|"  - сообщаем число срабатываний энкодера, которое осталось проехать
    "*OK_STOP|"          - подтверждаем получение команды остановиться
"""

import serial
import time

# PORT = '/dev/ttyS0'
# PORT = '/dev/ttyUSB0'
# PORT = 'COM4'

# при создании экземпляра класса, задаём порт и скорость обмена данными, для общения через UART.
class Arduino:
    def __init__(self, port, baudrate=115200, timeout=0, charset='utf-8'):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(4)  # ждём перезагрузку Arduino
        self.serial.reset_input_buffer()  # очищаем буфер входящих данных
        self.serial.reset_output_buffer()  # очищаем буфер отправляемых данных
        self.charset = charset  # запоминаем кодировку, с которой будем работать

        self.rxBuffer = ""  # переменная для хранения текста принимаемого сообщения

    # метод для отправки данных через UART
    def send_data(self, data: str):
        print('Sent to Arduino:', data)
        data = '*' + data + '|'  # добавляем символ конца строки
        self.serial.write(data.encode(self.charset))
        self.serial.flush()

    # метод для задания скорости
    def set_speed(self, speed):
        msg = f'SET_SPEED:{speed}'
        self.send_data(msg)
        self.serial.flush()

    # метод для задания угла поворота рулевых колёс
    def set_angle(self, angle):
        msg = f'SET_ANGLE:{angle}'
        self.send_data(msg)
        self.serial.flush()

    # метод для остановки беспилотника
    def stop(self):
        msg = 'STOP'
        self.send_data(msg)
        self.serial.flush()

    # метод для задания состояния светодиода
    def set_led(self, led_state):
        msg = f'SET_LED:{led_state}'
        self.send_data(msg)
        self.serial.flush()

    # метод для задания количества срабатываний энкодера до остановки
    def set_dist(self, dist):
        msg = f'SET_DIST:{dist}'
        self.send_data(msg)
        self.serial.flush()

    # запрос количества срабатываний энкодера до остановки
    def get_dist(self):
        msg = f'GET_DIST'
        self.send_data(msg)
        self.serial.flush()

    # метод, который корректно закрывает соединение
    # вызывать только в конце работы программы
    def close(self):
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.stop()  # останавливаем беспилотник
        time.sleep(1)  # пауза для доставки сообщения
        self.serial.close() # закрываем соединение

    # метод для получения сообщения от микроконтроллера
    # данные принимаются побайтно. Возвращает True, если сообщение получено целиком
    # в arduino.rxBuffer располагает текст сообщения без символов начала и конца сообщения
    def receive_message(self):
        while self.serial.in_waiting > 0:  # пока есть непрочитанные данные
            c = self.serial.read(1)  # читаем один байт
            c = c.decode(self.charset, errors='ignore')  # переводим байт в символ
            # игнорируем ошибки при переводе, чтобы программа работала дальше
            if c == '*':  # если нашли начало сообщения, то
                self.rxBuffer = ""  # обнуляем буфер, отбрасываем "битое" сообщение
                self.rxBuffer = self.rxBuffer + c
                continue  # переходим к чтению следующего символа

            # если буфер не пустой, значит мы уже нашли начало сообщения
            # только в этом случае мы записываем символы в буфер
            if self.rxBuffer != "":
                if c =='|':  # если нашли конец сообщения, то
                    # оставляем в буфере только текст полного сообщения без "*"
                    self.rxBuffer = self.rxBuffer[1:]
                    return True  # завершаем функцию и возвращаем True
                else:
                    self.rxBuffer += c  # добавляем символ к тексту сообщения

        # если цикл завершился, значит нет данных для чтения, завершаем функцию
        return False # возвращаем False, целого сообщения ещё нет

    # метод для получения уже принятого целого сообщения, готового для парсинга
    def get_msg_from_buffer(self):
        msg = self.rxBuffer  # запоминаем сообщение
        self.rxBuffer = ""  # очищаем буфер для записи следующего сообщения
        return msg  # возвращаем текст сообщения
