from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
import os
import pandas as pd

class Embedder:
    def __init__(self):
        self.path_array = []
        self.name = []

    def find_path(self, folder_path):
        with os.scandir(folder_path) as dirs:
            for entry in dirs:
                self.path_array.append(folder_path + entry.name)
                name = entry.name.split('.')
                self.name.append(name[0])

    def embedding(self, path):
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb([path])
        return embeddings

    def train_embedding(self, folder_path):
        self.find_path(folder_path)
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb(self.path_array)
            e = pd.DataFrame(embeddings)
            e["name"] = self.name
        return e