#!/usr/bin/env python3
"""Open and display an image with waitKey(0)."""

import sys
import cv2



image_path = "/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts_without_sync/task_4/image.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image '{image_path}'")
    sys.exit(1)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 33 px = 22.5 cm
# в 1 px  =  0.7 см