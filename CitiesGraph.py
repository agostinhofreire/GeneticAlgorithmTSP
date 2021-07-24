import numpy as np
import random
import pandas as pd
import sys


class CitiesGraph:

    def __init__(self, size=10):
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.size = size
        self.cities = self.labels[:self.size]

        if self.size > len(self.labels):
            print("[INFO] The number of cities must be less than", len(self.labels))
            sys.exit()

    def create(self):
        graph_up = np.random.randint(self.size**2, size=(self.size, self.size)) + 1
        np.fill_diagonal(graph_up, 0)


        for i in range(self.size):
            for j in range(self.size):
                if i < j:
                    graph_up[i, j] = 0

        graph_down = graph_up.T


        graph = graph_up + graph_down

        graph_df = pd.DataFrame(data=graph, index=self.cities, columns=self.cities)
        graph_df.to_csv("./cities.csv")

        return graph_df

    def load(self, path="./cities.csv"):
        return pd.read_csv(path, index_col=0)

    def getStart(self):
        return random.choice(self.cities)

