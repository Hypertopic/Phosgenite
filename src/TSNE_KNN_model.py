from embedder import Embedder
from rgb2hsv import Rgb2hsv

import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

'''
Class TSNE_KNN_model
This class create the model to predict the label of the input img

The model is the following:
- Input RGB image is transformed in HSV V
- Train and the input image are transformed in vectors of attributes with an embedder
- Train and input data are merged
- Train and input data are passed in TSNE to reduce dimsensions
- Train and input data are splited
- KNN (K = 1 euclidean metric) is trained with train data
- The label of the input data is predicted by the KNN
- The label is returned
'''
class TSNE_KNN_model:

    def __init__(self, img):
        self.cwd = os.getcwd()
        self.cwd.replace('/', '\\')
        self.test = img

        #use only if the dataset of images for the curch is not present in the folder dataset
        #self.transform_train()

        #use only if the dataset train_data_hsv_v.csv is present in the folder of the curch
        self.get_train()

        self.transform_test()
        self.tsne()
        lab = self.knn()

        return lab

    '''
    Transform train images
    convert RGB images in HSV V images and transform them in vectors of attributes with the embedder Painters
    '''
    def transform_train(self):
        Rgb2hsv().transform_folder(self.cwd + "/data/SJ/Vitraux baies/", self.cwd + "/data/SJ/Vitraux baies hsv/")
        self.train = Embedder().train_embedding(self.cwd + "/data/SJ/Vitraux baies hsv/")

    '''
    Get train dataset
    '''
    def get_train(self):
        self.train = pd.read_csv("./data/SJ/datasets/train_data_hsv_v.csv")

    '''
    Transform input image
    convert RGB image in HSV V image and transform it in a vector of attributes with the embedder Painters
    '''
    def transform_test(self):
        self.test = Rgb2hsv().transform_img(self.test, 'test.jpg')
        self.test = Embedder().embedding('test.jpg')

    '''
    Reduce dimensions of data using the TSNE
    '''
    def tsne(self):
        y_train = self.train["category"].values
        x_train = self.train.drop("category", axis=1).values

        # merge data
        X = np.concatenate([x_train, self.test])

        tsne = TSNE(n_components=2, perplexity=8, early_exaggeration=1)
        x = tsne.fit_transform(X)
        df = pd.DataFrame(x)

        #split data
        self.train = df[:(len(y_train))]
        self.test = df[(len(y_train)):]
        self.train["category"] = y_train

    '''
    Create KNN, train it with train data and predict the label of the input image
    '''
    def knn(self):
        # transform label (str) into number (int)
        le = preprocessing.LabelEncoder()
        self.train["category"] = le.fit_transform(self.train["category"])

        y_train = self.train["category"].values
        x_train = self.train.drop("category", axis=1).values
        test = self.test.values

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

        knn.fit(x_train, y_train)
        pred = knn.predict(test)

        # reverse the label transformation done at the begining of the function
        print("pred: ", le.inverse_transform(pred)[0])

        return le.inverse_transform(pred)[0]
