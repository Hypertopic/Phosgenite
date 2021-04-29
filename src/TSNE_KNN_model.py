from embedder import Embedder
from rgb2hsv import Rgb2hsv

import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


class TSNE_KNN_model:

    def __init__(self, img):
        self.cwd = os.getcwd()
        self.cwd.replace('/', '\\')
        self.test = img

    def transform_train(self):
        Rgb2hsv().transform_folder(self.cwd + "/data/SJ/Vitraux baies/", self.cwd + "/data/SJ/Vitraux baies hsv/")
        self.train = Embedder().train_embedding(self.cwd + "/data/SJ/Vitraux baies hsv/")
        print(self.train)

    def transform_test(self):
        self.test = Rgb2hsv().transform_img(self.test, 'test.png')
        self.test = Embedder().embedding('test.png')
        print(self.test)

    def tsne(self):
        y_train = self.train["name"].values
        x_train = self.train.drop("name", axis=1).values

        X = np.concatenate([x_train, self.test])
        print(pd.DataFrame(X))
        tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=1)
        x = tsne.fit_transform(X)
        df = pd.DataFrame(x)

        self.train = df[:(len(y_train))]
        self.test = df[(len(y_train)):]
        self.train["category"] = y_train

        print(self.train)


model = TSNE_KNN_model("C:/Users/antoi/Documents/UTT/ISI 4/PE/Phosgenite/src/data/SJ/test images/SJ 000 5.png")
model.transform_train()
model.transform_test()
model.tsne()



