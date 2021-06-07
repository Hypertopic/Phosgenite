from TSNE_KNN_model import TSNE_KNN_model

import time
import os

start = time.time()
TSNE_KNN_model(os.getcwd() + "/data/SJ/test images/SJ 000 1.jpg")
end = time.time()
print("Exdecution time: ", end - start)
