import cv2

class Cropper:

    #img must be the path to the img
    def __init__(self, img):
        self.img = img

    def crop(self):
        img = cv2.imread(self.img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray_img, 120, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        x = 10000000
        y = 0
        h = 10000000
        w = 0
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


        crop_img = img[h:y, x:w]
        cv2.imwrite("test.png", crop_img)

