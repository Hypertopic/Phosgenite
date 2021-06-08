from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
import os
import pandas as pd

'''
Class Embedder
Transform images in vectors of attributes using an the embedder Painters
Painters is available using ImageEmbedder from Orange 3 add on Image analytics
'''
class Embedder:
    def __init__(self):
        self.path_array = []
        self.category = []

    '''
    when transforming a folder of image this function find the path of images in the folder
    store each images path in an array
    '''
    def find_path(self, folder_path):
        with os.scandir(folder_path) as dirs:
            for entry in dirs:
                #store image path in array
                self.path_array.append(folder_path + entry.name)

                #store image name in array to have a list of category of stained glass present in the dataset
                name = entry.name.split('.')
                self.category.append(name[0])

    '''
    Use the embedder Painters to transform the image in input in a vector of attributes
    return the vector of attributes
    '''
    def embedding(self, path):
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb([path])
        return embeddings

    '''
    Use the embedder Painters to transform train images (contained in a folder) in vectors of attributes
    return an array of vectors of attributes
    '''
    def train_embedding(self, folder_path):
        self.find_path(folder_path)
        #loop on images path
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb(self.path_array)
            e = pd.DataFrame(embeddings)
            #add the target feature used to label the data
            e["category"] = self.category
        return e