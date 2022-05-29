from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sl

class RegresionLineal:

    def __init__(self, dataset, trainSize = 0.8):
        self.datasetOriginal = dataset.copy()
        
        limiteParticion = int(len(dataset) * trainSize)
        self.datasetTrain = dataset[0 : limiteParticion, :]
        self.datasetTest = dataset[limiteParticion : , : ]

    def AnalisisExploratorio(self):
        self.datasetOriginal.describe()

    def GenerarHistogramas(self):
        for columna in self.datasetOriginal.columns:
            sns.displot(self.datasetOriginal[columna])
