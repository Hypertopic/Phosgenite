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

        self.transform_train()
        self.transform_test()
        self.tsne()
        self.knn()

    def transform_train(self):
        Rgb2hsv().transform_folder(self.cwd + "/data/SJ/Vitraux baies/", self.cwd + "/data/SJ/Vitraux baies hsv/")
        self.train = Embedder().train_embedding(self.cwd + "/data/SJ/Vitraux baies hsv/")

    def transform_test(self):
        self.test = Rgb2hsv().transform_img(self.test, 'test.png')
        self.test = Embedder().embedding('test.png')

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

    def knn(self):
        le = preprocessing.LabelEncoder()
        self.train["category"] = le.fit_transform(self.train["category"])

        y_train = self.train["category"].values
        x_train = self.train.drop("category", axis=1).values
        test = self.test.values

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

        knn.fit(x_train, y_train)
        pred = knn.predict(self.test)

        print(pred)
        print("pred: ", le.inverse_transform(pred)[0])
