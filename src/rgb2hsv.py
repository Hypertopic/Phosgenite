import cv2
import os

'''
Class Rgb2hsv
This class allow to transform an image RGB or a folder of images RGB to images HSV
'''
class Rgb2hsv:

    def __init__(self):
        pass

    '''
    Transform the images RGB of the folder input in images HSV V
    Save images in folder_output
    '''
    def transform_folder(self, folder_input, folder_output):
        _, _, filenames = next(os.walk(folder_input))
        for file in filenames:
            # if path is not absolute
            '''
            abs_path = folder_input + file
            f = os.path.abspath(f)
            f.replace('/', '\\')
            '''
            img = cv2.imread(folder_input + file)
            # transform image in HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # save image HSV V in the output folder
            cv2.imwrite(folder_output + file, hsv[:, :, 2])


    '''
    Transform the file image RGB in an image HSV V and save it as output
    '''
    def transform_img(self, file, output):
        img = cv2.imread(file)

        # transform image in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # save image HSV V as output
        cv2.imwrite(output, hsv[:, :, 2])
