import cv2
import numpy as np
import datetime
import os

import pytesseract

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Detect the position of the windows
windows_list = []
toplist = []


def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))
    print(hwnd, win_text)


win32gui.EnumWindows(enum_win, toplist)
game_hwnd = 0
for (hwnd, win_text) in windows_list:
    if "Free Online Casino Games!" in win_text:
        game_hwnd = hwnd
while True:
    position = win32gui.GetWindowRect(game_hwnd)

    # Take screenshot
    screenshot = ImageGrab.grab(position)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Screen", screenshot)
    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    cv2.imwrite("Screenshots/" + moment + ".png", screenshot)
    key = cv2.waitKey(25)

    if key == 27:
        break


