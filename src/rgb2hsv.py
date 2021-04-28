import cv2
import os


class Rgb2hsv:

    def __init__(self):
        pass

    def transform_folder(self, folder_input, folder_output):
        _, _, filenames = next(os.walk(folder_input))

        os.chdir(folder_input)

        for file in filenames:
            img = cv2.imread(file)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            cv2.imwrite(folder_output + file, hsv[:, :, 2])

    def transform_img(self, file):
        img = cv2.imread(file)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return hsv[:, :, 2]
