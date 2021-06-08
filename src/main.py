from TSNE_KNN_model import TSNE_KNN_model

import time
import os

'''
Main function
call the function TSNE_KNN_model and pass the image to recognize
return the label of the image
'''
def find_label(img):
    start = time.time()
    lab = TSNE_KNN_model(os.getcwd() + img)
    end = time.time()
    print("Exdecution time: ", end - start)
    return lab


#exemple of use
find_label("/data/SJ/test images/SJ 000 1.jpg")