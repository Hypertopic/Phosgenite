import cv2

'''
Class cropper
This class use the function FindContours of CV2 to detect contours of an images
The image is cropped using the bottom left corner and top right corner
'''
class Cropper:

    #img must be the path to the img
    def __init__(self, img):
        self.img = img

    '''
    Crop the image passed in the creation of the class
    '''
    def crop(self):
        img = cv2.imread(self.img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set threshold to 120 to avoid noise caused by luminosity
        retval, thresh = cv2.threshold(gray_img, 120, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #x, y: coordinates of the bottom left corner of the image
        # h, w: height and weight of the image

        # set x to a max value to find the lowest x possible
        x = 10000000
        y = 0
        # set h to a max value to find the lowest h possible
        h = 10000000
        w = 0

        #search mimimum and maximum points in contours
        for con in contours:
            for co in con:
                for c in co:
            #bootom left
                    if c[0] < x:
                        x = c[0]
                    if c[1] > y:
                        y = c[1]
            #top right
                    if c[0] > w:
                        w = c[0]
                    if c[1] < h:
                        h = c[1]

        #crop image
        crop_img = img[h:y, x:w]
        cv2.imwrite("test.png", crop_img)

