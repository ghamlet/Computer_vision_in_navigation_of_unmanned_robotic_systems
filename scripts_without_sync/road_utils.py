# coding: utf-8
from math import sin, cos, radians

import cv2
import numpy as np

KP = 0.55  # 0.22 0.32 0.42
KD = 0.15  # 0.17
last = 0

SIZE = (533, 300)

RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])

src_draw = np.array(TRAP, dtype=np.int32)


THRESHOLD = 220
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def wrap(image, angle=0):
    if not (0 <= angle < 90):
        raise RuntimeError('Angle should be [0, 90)')
    image = image.copy()
    h, w = image.shape[:2]
    orig = np.float32(
        [[0, h], # left - bottom 
        [w, h],  # right - bottom
        [w, 0],  # right - up
        [0, 0]]  # left - up
    )
    h_new = h * cos(radians(angle))
    w_offset = (w//2)* sin(radians(angle))

    image = image[int((h - h_new) * 1.2):, :]

    dist = np.float32(
        [[0   , h],
        [w, h],
        [w - w_offset, 0],
        [w_offset, 0]]
    )
    M = cv2.getPerspectiveTransform(dist, orig)
    perspective = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return perspective

def find_lines2(frame):
    src = np.float32([[20, 200],  # точки исходной трапеции
                      [350, 200],
                      [275, 80],
                      [85, 80]])
    img_size = [200, 360]

    src_draw = np.array(src, dtype=np.int32)
    dst = np.float32([[0, img_size[0]],        # точки получающегося прямоугольника
                      [img_size[1], img_size[0]],
                      [img_size[1], 0],
                      [0, 0]])
    resized = cv2.resize(frame, (img_size[1], img_size[0]))
    #cv2.imshow("frame", resized)

    # проводим бинаризацию
    r_channel = resized[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 200)] = 1
    # cv2.imshow("r_channel",binary)

    hls = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
    s_channel = resized[:, :, 2]
    binary2 = np.zeros_like(s_channel)
    binary2[(r_channel > 160)] = 1

    allBinary = np.zeros_like(binary)
    allBinary[((binary == 1) | (binary2 == 1))] = 255
    #cv2.imshow("binary", allBinary)

    # проводим преобразование изображения к виду, как будто смотрим на него сверху
    allBinary_visual = allBinary.copy()
    cv2.polylines(allBinary_visual, [src_draw], True, 255)  # отображаем исходную трапецию
    #cv2.imshow("polygon", allBinary_visual)

    M = cv2.getPerspectiveTransform(src,
                                    dst)  # рассчитываем матрицу для преобразования исходной трапеции в прямоугольник
    warped = cv2.warpPerspective(allBinary, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
    #cv2.imshow("warped", warped)  # отображаем преобразованное изображение

    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)  # Считаем в нижней половине суммы столбцов

    midpoint = histogram.shape[0] // 2
    IndWhitestColumnL = np.argmax(histogram[:midpoint])  # Считаем координаты самого белого столбца слева
    IndWhitestColumnR = np.argmax(histogram[midpoint:]) + midpoint  # Считаем координаты самого белого столбца справа
    warped_visual = warped.copy()  # отображаем их
    cv2.line(warped_visual, (IndWhitestColumnL, 0), (IndWhitestColumnL, warped_visual.shape[0]), 110, 2)
    cv2.line(warped_visual, (IndWhitestColumnR, 0), (IndWhitestColumnR, warped_visual.shape[0]), 110, 2)
    #cv2.imshow("WitestColumn", warped_visual)

    # А теперь будем рулить, давайте на кадре, будем писать, куда подруливаем.

    # cv2.putText(frame, "Text", (object[2][0], object[2][1]-20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, [100, 255, 0], 1)

    nwindows = 9
    window_height = int(warped.shape[0] / nwindows)
    window_half_width = 25

    XCenterLeftWindow = IndWhitestColumnL
    XCenterRightWindow = IndWhitestColumnR

    left_lane_inds = np.array([], dtype=np.int16)
    right_lane_inds = np.array([], dtype=np.int16)

    out_img = np.dstack((warped, warped, warped))

    nonzero = warped.nonzero()
    WhitePixelIndY = np.array(nonzero[0])
    WhitePixelIndX = np.array(nonzero[1])

    for window in range(nwindows):

        win_y1 = warped.shape[0] - (window + 1) * window_height
        win_y2 = warped.shape[0] - (window) * window_height

        left_win_x1 = XCenterLeftWindow - window_half_width
        left_win_x2 = XCenterLeftWindow + window_half_width
        right_win_x1 = XCenterRightWindow - window_half_width
        right_win_x2 = XCenterRightWindow + window_half_width

        cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
        cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21), 2)
        # cv2.imshow("windows", out_img)

        good_left_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                          (WhitePixelIndX >= left_win_x1) & (WhitePixelIndX <= left_win_x2)).nonzero()[0]

        good_right_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                           (WhitePixelIndX >= right_win_x1) & (WhitePixelIndX <= right_win_x2)).nonzero()[0]

        left_lane_inds = np.concatenate((left_lane_inds, good_left_inds))
        right_lane_inds = np.concatenate((right_lane_inds, good_right_inds))

        if len(good_left_inds) > 50:
            XCenterLeftWindow = int(np.mean(WhitePixelIndX[good_left_inds]))
        if len(good_right_inds) > 50:
            XCenterRightWindow = int(np.mean(WhitePixelIndX[good_right_inds]))

    cnt = np.array(list(zip(WhitePixelIndX[left_lane_inds], WhitePixelIndY[left_lane_inds])) + list(
        zip(WhitePixelIndX[right_lane_inds], WhitePixelIndY[right_lane_inds]))[::-1])
    if not cnt.shape[0]:
        return np.zeros(frame.shape[:2])
    print(cv2.boundingRect(cnt))
    # cnt =cv2.convexHull(cnt)
    print(cnt.shape)
    mask = np.zeros(img_size,dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    M = cv2.getPerspectiveTransform(dst,
                                    src)  # рассчитываем матрицу для преобразования исходной трапеции в прямоугольник
    mask = cv2.warpPerspective(mask, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
    mask =cv2.resize(mask, frame.shape[:2][::-1])
    return mask

def binarize(img, threshold, show=False):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    binary_h = cv2.inRange(hls, (0, 0, 30), (255, 255, 255))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_g = cv2.inRange(gray, threshold, 255)  # 130

    binary = cv2.bitwise_and(binary_g, binary_h)

    if show:
        cv2.imshow('hls', hls)
        cv2.imshow('hlsRange', binary_h)
        cv2.imshow('grayRange', binary_g)
        cv2.imshow('gray', gray)
        cv2.imshow('bin', binary)

    return binary


def binarize_exp(img, d=0):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_s_channel = hls[:, :, 2]
    hls_l_channel = hls[:, :, 1]
    hls_h_channel = hls[:, :, 0]
    hsv_h_channel = hsv[:, :, 2]
    hsv_s_channel = hsv[:, :, 1]
    hsv_v_channel = hsv[:, :, 0]
    binary_h = cv2.inRange(hls, (0, 0, 30), (255, 255, 205))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_g = cv2.inRange(gray, 130, 255) #130
    binary = cv2.bitwise_and(binary_g, binary_h)

    if d:
        cv2.imshow('hls', hls)
        cv2.imshow('bgr_b', img[:, :, 0])
        cv2.imshow('bgr_g', img[:, :, 1])
        cv2.imshow('bgr_r', img[:, :, 2])
        cv2.imshow('hls_s', hls_s_channel)
        cv2.imshow('hls_l', hls_l_channel)
        cv2.imshow('hls_h', hls_h_channel)
        cv2.imshow('hsv_h', hsv_h_channel)
        cv2.imshow('hsv_s', hsv_s_channel)
        cv2.imshow('hsv_v', hsv_v_channel)
        cv2.imshow('hlsRange', binary_h)
        cv2.imshow('grayRange', binary_g)
        cv2.imshow('gray', gray)
        cv2.imshow('bin', binary)

    # return binary
    return binary

def trans_perspective(binary, trap, rect, size, d=0):
    matrix_trans = cv2.getPerspectiveTransform(trap, rect)
    perspective = cv2.warpPerspective(binary, matrix_trans, size, flags=cv2.INTER_LINEAR)
    if d:
        cv2.imshow('perspective', perspective)
    return perspective


def find_left_right(perspective, show=False):
    h, w = perspective.shape[:2]
    hist = np.sum(perspective[perspective.shape[0] // 3:, :], axis=0)
    mid = hist.shape[0] // 2
    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid
    if left <= 10 and right - mid <= 10:
        right = 399

    if show:
        cv2.line(perspective, (left, 0), (left, h), 50, 2)
        cv2.line(perspective, (right, 0), (right, h), 50, 2)
        cv2.line(perspective, ((left + right) // 2, 0), ((left + right) // 2, h), 110, 3)
        cv2.imshow('lines', perspective)

    return left, right

def find_left_right2(perspective, show=False):
    perspective = perspective.copy()
    perspective_draw = perspective.copy()

    perspective = perspective[perspective.shape[0] // 3:, :]
    h, w  = perspective.shape[:2]
    hist  = np.sum(perspective, axis=0)
    mid   = hist.shape[0] // 2
    left  = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid

    left_amount  = hist[left]//255
    right_amount = hist[right]//255
    left_side_amount  = np.sum(hist[:mid])//255
    right_side_amount = np.sum(hist[mid:])//255
    print("---AMOUNT---")
    print(f"left:  {left_amount:3}/{h:3} ({left_amount/h:.2f})")
    print(f"right: {right_amount:3}/{h:3} ({right_amount/h:.2f})")
    print(f"left side: {left_side_amount:4}/{ h*mid:4} ({left_side_amount/ (h*mid):.2f})")
    print(f"left side: {right_side_amount:4}/{h*mid:4} ({right_side_amount/(h*mid):.2f})")
    print("------------\n")

    left_found = True
    right_found = True
    
    if abs(mid - left) < 0.05 * w or left_side_amount  < h*mid*0.005:
        left = 0
        left_found = False

    if abs(mid - right) < 0.05 * w or right_side_amount < h*mid*0.015:
        right = w
        right_found = False

    if show:
        cv2.line(perspective_draw, (left, 0), (left, h), 50, 2)
        cv2.line(perspective_draw, (right, 0), (right, h), 50, 2)
        cv2.line(perspective_draw, ((left + right) // 2, 0), ((left + right) // 2, h), 110, 3)
        cv2.imshow('lines', perspective_draw)

    find_left_right2.left_amount = left_amount/h
    find_left_right2.left_side_amount = left_side_amount/ (h*mid)
    find_left_right2.left_found = left_found

    find_left_right2.right_amount = right_amount/h
    find_left_right2.right_side_amount = right_side_amount/ (h*mid)
    find_left_right2.right_found = right_found

    return left, right


def centre_mass(perspective, d=0):
    hist = np.sum(perspective, axis=0)
    if d:
        cv2.imshow("Perspektiv2in", perspective)

    mid = hist.shape[0] // 2
    i = 0
    centre = 0
    sum_mass = 0
    while (i <= mid):
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_left = centre / sum_mass
    else:
        mid_mass_left = mid-1

    centre = 0
    sum_mass = 0
    i = mid
    while i < hist.shape[0]:
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_right = centre / sum_mass
    else:
        mid_mass_right = mid+1

    # print(mid_mass_left)
    # print(mid_mass_right)
    mid_mass_left = int(mid_mass_left)
    mid_mass_right = int(mid_mass_right)
    
    if d:
        cv2.line(perspective, (mid_mass_left, 0), (mid_mass_left, perspective.shape[1]), 50, 2)
        cv2.line(perspective, (mid_mass_right, 0), (mid_mass_right, perspective.shape[1]), 50, 2)
        # cv2.line(perspective, ((mid_mass_right + mid_mass_left) // 2, 0), ((mid_mass_right + mid_mass_left) // 2, perspective.shape[1]), 110, 3)
        cv2.imshow('CentrMass', perspective)

    return mid_mass_left, mid_mass_right


def centre_mass2(perspective, d=0):
    hist = np.sum(perspective, axis=0)
    h, w = perspective.shape[:2]
    if d:
        cv2.imshow("Perspektiv2in", perspective)

    mid = hist.shape[0] // 2
    i = 0
    centre = 0
    sum_mass = 0
    while (i <= mid):
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1

    if sum_mass > 0:
        mid_mass_left = centre / sum_mass
        if abs(mid - mid_mass_left) < 0.05 * w:
            left_found = False
        else:
            left_found = True
    else:
        left_found = False
    
    if not left_found:
        mid_mass_left = w//3

    i = mid
    centre = 0
    sum_mass = 0
    while i < hist.shape[0]:
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_right = centre / sum_mass
        if abs(mid - mid_mass_right) < 0.05 * w:
            right_found = False
        else:
            right_found = True
    else:
        right_found = False

    if not right_found:
        mid_mass_right = w - 1

    # print(mid_mass_left)
    # print(mid_mass_right)
    mid_mass_right = min(w - 1, mid_mass_right)
    mid_mass_left = int(mid_mass_left)
    mid_mass_right = int(mid_mass_right)

    left_amount  = hist[mid_mass_left]//255
    right_amount = hist[mid_mass_right]//255
    left_side_amount  = np.sum(hist[:mid])//255
    right_side_amount = np.sum(hist[mid:])//255
    # print("---AMOUNT---")
    # print(f"left:  {left_amount:3}/{h:3} ({left_amount/h:.2f})")
    # print(f"right: {right_amount:3}/{h:3} ({right_amount/h:.2f})")
    # print(f"left side: {left_side_amount:4}/{ h*mid:4} ({left_side_amount/ (h*mid):.2f})")
    # print(f"left side: {right_side_amount:4}/{h*mid:4} ({right_side_amount/(h*mid):.2f})")
    # print("------------\n")

    centre_mass2.left_amount = left_amount/h
    centre_mass2.left_side_amount = left_side_amount/ (h*mid)
    centre_mass2.left_found = left_found

    centre_mass2.right_amount = right_amount/h
    centre_mass2.right_side_amount = right_side_amount/ (h*mid)
    centre_mass2.right_found = right_found


    if d:
        cv2.line(perspective, (mid_mass_left, 0), (mid_mass_left, perspective.shape[1]), 50, 2)
        cv2.line(perspective, (mid_mass_right, 0), (mid_mass_right, perspective.shape[1]), 50, 2)
        # cv2.line(perspective, ((mid_mass_right + mid_mass_left) // 2, 0), ((mid_mass_right + mid_mass_left) // 2, perspective.shape[1]), 110, 3)
        cv2.imshow('CentrMass', perspective)

    return mid_mass_left, mid_mass_right


def detect_stop(perspective):
    hist = np.sum(perspective, axis=1)
    maxStrInd = np.argmax(hist)
    # print("WhitePixCual" + str(hist[maxStrInd]//255))
    if hist[maxStrInd]//255 > 150:
        # print("SL detected. WhitePixselCual: "+str(int(hist[maxStrInd]/255)) + "Ind: " + str(maxStrInd))
        if maxStrInd > 120:  # 100
            # print("Time to stop")
            # cv2.line(perspective, (0, maxStrInd), (perspective.shape[1], maxStrInd), 60, 4)
            # cv2.imshow("STOP| ind:"+str(maxStrInd)+"IndCual"+str(hist[maxStrInd]//255), perspective)
            return True
    return False


def detect_stop2(perspective, white_limit = 0.25, show = False, text = False):
    h, w = perspective.shape[:2]
    rows_brigthness = np.sum(perspective, axis=1)
    brightest_row_id = np.argmax(rows_brigthness)
    row_white_pixel_amount = rows_brigthness[brightest_row_id]//255

    if text:
        print(f"Stop line: {row_white_pixel_amount:3} > {int(w * white_limit)} is {row_white_pixel_amount > w * white_limit}; " \
              f"row: {brightest_row_id:3} >= {h // 3} is {brightest_row_id >= h // 3}")

    if show:
        stop_frame = perspective.copy()
        cv2.line(stop_frame, (0, brightest_row_id), (w, brightest_row_id), 255)
        cv2.imshow('stop', stop_frame) 

    # если в самой яркой строке больше white_limit процентов белых пикселей и строка ниже середины изображения
    if row_white_pixel_amount > w * white_limit and brightest_row_id >= h//3:
        return True
    return False


def detect_left_turn(bin):
    bin = bin.copy()
    bin = np.flip(bin, axis=0)[:, bin.shape[1]//2:]
    hist = np.argmax(bin, axis=1)
    idx = np.argsort(hist)[-1]
    # print(idx)
    if idx < bin.shape[0]//3:
        return True
    return False

def detect_straight_again(bin):
    pixel_amount = int(0.1 * bin.shape[1])
    bin = np.flip(bin, axis=0)[bin.shape[0]//3:]
    hist = np.argmax(bin, axis=0)
    ind_max_elem = np.argsort(hist)[-pixel_amount:]
    left = right = 0
    for idx in ind_max_elem:
        if idx < bin.shape[1]//2:
            left += 1
        else:
            right += 1
    return right == 0

def detect_return_road(wrapped: np.ndarray, left_side_amount, right_side_amount, check_points=[0.3, 0.5, 0.9], show=False, draw_on_input=False):
    h, w = wrapped.shape[:2]
    mid_w = w//2

    check_points_abs = [int(p*h) for p in check_points]
    hist_left: np.ndarray = np.sum(wrapped[:, :mid_w], axis=1)//255
    hist_right: np.ndarray = np.sum(wrapped[:, mid_w:], axis=1)//255

    # проверяем есть ли хотя бы один пиксель на каждом из уровней check_points слева и справа
    conds = [hist_left[p] >= 1 for p in check_points_abs] + [hist_right[p] >= 1 for p in check_points_abs]
    # print(conds)
    #print(conds, left_side_amount)
    if show:
        img_draw = wrapped.copy()

        for y in check_points_abs:
            cv2.line(img_draw, (0, y), (w, y), (255, 255, 255), 1)

        cv2.imshow('detect_road_return', img_draw)


    if all(conds) and left_side_amount > 0.005:
        return True
    return False
    


def cross_center_path_v4_2(bin, pixel_offset=0, line_amount_percent=0.1, bottom_offset_percent=0.3, show_all_lines=False): 
    if pixel_offset:
        bin = bin[:, pixel_offset:-pixel_offset]
    
    bin = np.flip(bin, axis=0)[int(bin.shape[0]*bottom_offset_percent):]
    bin[-1] = 255
    line_amount = int(line_amount_percent * bin.shape[1])

    hist = np.argmax(bin, axis=0)
    ind_max_elems = np.argsort(hist)[-line_amount:]
    avg_idx = int(np.mean(ind_max_elems))
    max_dist = hist[avg_idx]/bin.shape[0]
    if show_all_lines:
        black_lines = bin.copy()
        left = right = 0
        for idx in ind_max_elems:
            if idx < black_lines.shape[1]//2:
                left += 1
            else:
                right += 1
            line_value = hist[idx]
            cv2.line(black_lines, (idx, 0), (idx, line_value), 150)
        # print('left:', left)
        # print('right:', right)
        # print('find turn:', right == 0)
        black_lines = np.flip(black_lines, axis=0)
        cv2.imshow('black_lines', black_lines)
    return avg_idx + pixel_offset, max_dist


def cross_center_path_v1(bin):
    hist = np.zeros((2, bin.shape[1]), dtype=np.int32)
    begin_black_part = 0
    len_black_part = 0

    for i in range(bin.shape[1]):  # перебираем столбцы
        for j in range(bin.shape[0]):  # перебираем строки
            if bin[j, i] == 255:
                if len_black_part > hist[1, i]:  # max_len_black_part:
                    hist[1, i] = len_black_part
                    hist[0, i] = begin_black_part
                len_black_part = 0
                begin_black_part = j
                print(begin_black_part is j)
                # print("***")
                # print(begin_black_part, len_black_part)
                # print(begin_max_black_part, max_len_black_part)
                # print("***")
            else:
                len_black_part += 1
        if len_black_part > hist[1, i]:
            hist[1, i] = len_black_part
            hist[0, i] = begin_black_part

        begin_black_part = 0
        len_black_part = 0

    # рисую самые длинные чёрные отрезки
    bin_viz = bin.copy()
    for i in range(bin.shape[1]):
        bin_viz[hist[0, i]:(hist[0, i]+hist[1, i]), i] = 100
    cv2.imshow("cross_path", bin_viz)

    return False


def cross_center_path_v2(bin):
    hist = np.zeros(bin.shape[1], dtype=np.int32)
    bin_viz = bin.copy()
    for i in range(bin.shape[1]):  # columns
        for j in range(bin.shape[0] - 1, -1, -1):  # string
            if bin_viz[j, i] == 255:
                bin_viz[:j, i] = 255
                hist[i] = j
                break
            else:
                bin_viz[j, i] = 100
    cv2.imshow("cross_path", bin_viz)

    return False

def cross_center_path_v3(bin):
    """
    Для каждого столбца выбирает координату самого нижнего белого пикселя.
    Выбирает 30 столбцов, в которых белый пиксель максимально высоко.
    Рулит на среднее арифметическое от координат этих столбцов.
    """
    bin_viz = bin.copy()
    bin = np.flip(bin, axis=0)
    cv2.imshow("bin_invert", bin)
    hist = np.argmax(bin, axis=0)
    hist = bin.shape[0] - hist

    for i in range(hist.shape[0]):
        bin_viz[hist[i]:, i] = 100
    cv2.imshow("cross_path", bin_viz)

    hist = bin.shape[0] - hist
    ind_max_elem = np.argsort(hist)
    ind_max_elem = ind_max_elem[-30:]
    for i in ind_max_elem:
        bin_viz[:, i] = 50
    #cv2.imshow("cross_path_pulling", bin_viz)

    err = (bin.shape[1] // 2) - int(np.mean(ind_max_elem))

    bin_viz[:, bin.shape[1]//2] = 255
    cv2.line(bin_viz, (bin.shape[1]//2, bin.shape[0]), (bin.shape[1]//2-err, bin.shape[0]//2), 255, 4)
    cv2.imshow("cross_path_3", bin_viz)

    return err


def cross_center_path_v4(bin):  # таже третья, но без мишуры и визуализации
    """
        Для каждого столбца выбирает координату самого нижнего белого пикселя.
        Выбирает 30 столбцов, в которых белый пиксель максимально высоко.
        Рулит на среднее арифметическое от координат этих столбцов.
    """
    bin = np.flip(bin, axis=0)
    hist = np.argmax(bin, axis=0)
    ind_max_elem = np.argsort(hist)[-30:]
    err = (bin.shape[1] // 2) - int(np.mean(ind_max_elem))
    return err


def cross_center_path_v5(bin): # дорабатываем v3, чтобы рулило на всех наших перекрёстках.
    """
    Для каждого столбца выбирает координату самого нижнего белого пикселя.
    Выбирает Х (pull_size) столбцов, в которых белый пиксель максимально высоко.
    Х (pull_size) - зависит от отношения максимального к среднему.
    Рулит на среднее арифметическое от координат этих столбцов.
    """
    bin_viz = bin.copy()
    bin = np.flip(bin, axis=0)
    cv2.imshow("bin_invert", bin)
    hist = np.argmax(bin, axis=0)
    hist = bin.shape[0] - hist

    for i in range(hist.shape[0]):
        bin_viz[hist[i]:, i] = 100
    cv2.imshow("cross_path", bin_viz)

    hist = bin.shape[0] - hist

    max_columh = np.max(hist)
    mean_column = np.mean(hist)
    pull_size = max_columh / mean_column
    print(max_columh)
    print(mean_column)
    print(pull_size)
    print("*----")
    if pull_size >= 1.6:  # 1.6
        pull_size = int(50 * pull_size)
    else:
        pull_size = 30
    ind_max_elem = np.argsort(hist)
    ind_max_elem = ind_max_elem[-pull_size:]
    for i in ind_max_elem:
        bin_viz[:, i] = 50
    #cv2.imshow("cross_path_pulling", bin_viz)

    err = (bin.shape[1] // 2) - int(np.mean(ind_max_elem))

    bin_viz[:, bin.shape[1]//2] = 255
    cv2.line(bin_viz, (bin.shape[1]//2, bin.shape[0]), (bin.shape[1]//2-err, bin.shape[0]//2), 255, 4)
    cv2.imshow("cross_path_5", bin_viz)

    return err


def cross_center_path_v6(bin):  # таже пятая, но без мишуры и визуализации
    """
        Для каждого столбца выбирает координату самого нижнего белого пикселя.
        Выбирает 30 столбцов, в которых белый пиксель максимально высоко.
        Рулит на среднее арифметическое от координат этих столбцов.
    """
    bin = np.flip(bin, axis=0)
    hist = np.argmax(bin, axis=0)

    pull_size = np.max(hist) / np.mean(hist)
    if pull_size >= 1.6:  # 1.6
        pull_size = int(50 * pull_size)
    else:
        pull_size = 30

    ind_max_elem = np.argsort(hist)[-pull_size:]
    err = (bin.shape[1] // 2) - int(np.mean(ind_max_elem))
    return err




def detect_road_begin(perspective):  # для переключения с пересеченипея перекрёстка на следование по разметке
    left_corner = np.sum(perspective[-50:, :perspective.shape[1] // 3])
    right_corner = np.sum(perspective[-50:, perspective.shape[1] // 3 * 2:])
    print(left_corner)
    print(right_corner)
    print("**----**")
    # if left_corner >= 500000 and right_corner >= 170000:
    # if left_corner >= 170000 and right_corner >= 170000:
    if left_corner >= 170000 and right_corner >= 170000:
        return True
    else:
        return False