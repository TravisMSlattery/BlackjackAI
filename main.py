import cv2
import numpy as np
import time
import os

# Detect the position of the windows
import win32gui
from PIL import ImageGrab
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
windows_list = []
toplist = []


def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))
    print(hwnd, win_text)


win32gui.EnumWindows(enum_win, toplist)
game_hwnd = 0
for (hwnd, win_text) in windows_list:
    if "247 Blackjack - Google Chrome" in win_text:
        game_hwnd = hwnd
while True:
    position = win32gui.GetWindowRect(game_hwnd)

    # Take screenshot
    image = ImageGrab.grab(position)
    image = np.array(image)
    #pre processing for tesseract
    # get grayscale image
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # noise removal
    def remove_noise(image):
        return cv2.medianBlur(image, 5)


    # thresholding
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    # dilation
    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)


    # erosion
    def erode(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)


    # opening - erosion followed by dilation
    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


    # canny edge detection
    def canny(image):
        return cv2.Canny(image, 100, 200)


    # skew correction
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

# template matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    ##Detecting Characters
    (hImag,wImg) = image.shape
    boxes = pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        print(b)
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.retangle(image,(x))
    #cv2.imshow("Screen", screenshot)
    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    cv2.imwrite("Screenshots/" + moment + ".png", image)
    key = cv2.waitKey(1000)


    if key == 27:
            break
cv2.destroyAllWindows()

