# Задача 5: детекция светофора (working / notworking)

Обученная и квантованная модель YOLOv4-tiny для распознавания состояния светофора.

## Классы модели

| # | Имя        | Описание                          |
|---|------------|-----------------------------------|
| 0 | `notworking` | неисправный / погашенный светофор |
| 1 | `working`    | рабочий светофор                  |

Список классов: `model/svetofor.names`.

## Состав папки

```
task_5/
├── model/
│   ├── yolov4-tiny-svetofor.cfg                   # архитектура сети (Darknet, 416×416)
│   ├── yolov4-tiny-svetofor_best_weights.weights  # веса FP32 (Darknet) ~22 МБ
│   ├── yolov4-tiny-svetofor_best_weights_uint8.tmfile  # квантованная модель uint8 (Tengine) ~5.6 МБ
│   ├── svetofor.names                             # имена классов
│   └── svetofor.data                              # служебный файл Darknet (пути из Colab)
├── dt/                                            # калибровочный датасет (100 фото, для квантования)
├── detect_svetofor.py                             # быстрый тест модели через OpenCV DNN
└── README.md
```

## Как использовать модель

### 1. На ПК (быстрый тест, OpenCV DNN)

Работает с весами FP32 (`*.weights`), не требует робота и Tengine. Подойдёт для проверки качества детекции на видео:

```bash
python3 detect_svetofor.py --source ../records/received_20260731_172046.avi
```

Опции скрипта:

```bash
python3 detect_svetofor.py \
  --source video.avi      # или --source 0 (камера), --source http://...:8081/
  --conf 0.25             # порог уверенности
  --nms 0.4               # порог NMS
  --skip 0                # пропускать N кадров между детекциями (ускоряет)
  --save result.avi       # сохранить размеченное видео
```

По умолчанию `--source auto` берёт первый ролик из `records/`. Управление в окне: `space` — пауза, `ESC`/`q` — выход.

### 2. На роботе (квантованная uint8 модель, yolopy + TIM-VX)

Файл `model/yolov4-tiny-svetofor_best_weights_uint8.tmfile` — формат Tengine, запускается через библиотеку `yolopy` на NPU (TIM-VX). Пример:

```python
import cv2
import yolopy

model_file = 'model/yolov4-tiny-svetofor_best_weights_uint8.tmfile'
classes = ['notworking', 'working']

model = yolopy.Model(model_file, use_uint8=True, use_timvx=True, cls_num=2)
# якоря из model/yolov4-tiny-svetofor.cfg (строки anchors)
model.set_anchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    classes_ids, scores, boxes = model.detect(frame)
    for cls_id, score, box in zip(classes_ids, scores, boxes):
        if score > 0.3:
            label = classes[cls_id]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('svetofor', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
```

**Важно:**
- Параметр `cls_num=2` должен совпадать с числом классов в `svetofor.names` (иначе yolopy вернёт неверные классы).
- Якоря `set_anchors` берутся из `model/yolov4-tiny-svetofor.cfg` (строка `anchors`), порядок как в cfg.
- Вход модели 416×416, кадры приводятся к этому размеру автоматически внутри yolopy.

## Как была получена uint8-модель

Квантование выполнено инструментом Tengine (`quant_tool.py` из `Bazovyi_774_kod_-_AI_774_KAR/`):

```bash
python3 quant_tool.py \
  -w model/yolov4-tiny-svetofor_best_weights.weights \
  -c model/yolov4-tiny-svetofor.cfg \
  -d dt
```

1. `convert_tool` конвертирует Darknet (`.cfg` + `.weights`) в FP32 `.tmfile`.
2. `quant_tool_uint8` квантует FP32-модель в uint8 по калибровочным изображениям из `dt/`
   (вход `3,416,416`, scale `1/255`, mean `0,0,0`, letterbox `416,416`, 8 потоков).
3. На выходе — `yolov4-tiny-svetofor_best_weights_uint8.tmfile`.

Для повторного квантования бинарники Tengine должны лежать в `~/Tengine/build/install/bin/`
(`convert_tool` и `quant_tool_uint8`). Калибровочные фото — просто изображения с видимыми
светофорами, разметка не нужна.
