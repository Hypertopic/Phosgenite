from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
import os
import pandas as pd

class Embedder:
    def __init__(self, path):
        self.path = path
        self.path_array = []
        self.name = []

    def find_path(self):
        with os.scandir(self.path) as dirs:
            for entry in dirs:
                self.path_array.append(self.path + entry.name)

    def train_find_path(self):
        with os.scandir(self.path) as dirs:
            for entry in dirs:
                self.path_array.append(self.path + entry.name)
                name = entry.name.split('.')
                self.name.append(name[0])

    def embedding(self):
        self.find_path()
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb(self.path_array)
        return embeddings


    def train_embedding(self):
        self.train_find_path()
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb(self.path_array)
            e = pd.DataFrame(embeddings)
            e["name"] = self.name
        return e