# coding: utf-8
"""Утилиты компьютерного зрения для поиска дорожной разметки.

Урезанная версия road_utils.py из Bazovyi_774_kod_-_AI_774_KAR:
без констант и функций управления колёсами (KP, KD) — только зрение.
"""

import cv2
import numpy as np

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


def trans_perspective(binary, trap, rect, size, d=0):
    matrix_trans = cv2.getPerspectiveTransform(trap, rect)
    perspective = cv2.warpPerspective(binary, matrix_trans, size, flags=cv2.INTER_LINEAR)
    if d:
        cv2.imshow('perspective', perspective)
    return perspective


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

    mid_mass_right = min(w - 1, mid_mass_right)
    mid_mass_left = int(mid_mass_left)
    mid_mass_right = int(mid_mass_right)

    left_amount = hist[mid_mass_left]//255
    right_amount = hist[mid_mass_right]//255
    left_side_amount = np.sum(hist[:mid])//255
    right_side_amount = np.sum(hist[mid:])//255

    centre_mass2.left_amount = left_amount/h
    centre_mass2.left_side_amount = left_side_amount / (h*mid)
    centre_mass2.left_found = left_found

    centre_mass2.right_amount = right_amount/h
    centre_mass2.right_side_amount = right_side_amount / (h*mid)
    centre_mass2.right_found = right_found

    if d:
        cv2.line(perspective, (mid_mass_left, 0), (mid_mass_left, perspective.shape[1]), 50, 2)
        cv2.line(perspective, (mid_mass_right, 0), (mid_mass_right, perspective.shape[1]), 50, 2)
        cv2.imshow('CentrMass', perspective)

    return mid_mass_left, mid_mass_right
