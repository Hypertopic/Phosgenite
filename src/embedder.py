from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
import os


class Embedder:
    def __init__(self, path):
        self.path = path
        self.path_array = []

    def find_path(self):
        with os.scandir(self.path) as dirs:
            for entry in dirs:
                self.path_array.append(self.path + entry.name)

    def embedding(self):
        self.find_path()
        with ImageEmbedder(model='painters') as emb:
            embeddings = emb(self.path_array)
        return embeddings
