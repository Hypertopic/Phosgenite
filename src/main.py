from TSNE_KNN_model import TSNE_KNN_model

import time
import os

def find_label(img):
    start = time.time()
    lab = TSNE_KNN_model(os.getcwd() + img)
    end = time.time()
    print("Exdecution time: ", end - start)
    return lab


#exemple
find_label("/data/SJ/test images/SJ 000 1.jpg")