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
        graph = np.random.randint(self.size**2, size=(self.size, self.size)) + 1
        np.fill_diagonal(graph, 0)
        graph_df = pd.DataFrame(data=graph, index=self.cities, columns=self.cities)

        return graph_df

    def getStart(self):
        return random.choice(self.cities)

