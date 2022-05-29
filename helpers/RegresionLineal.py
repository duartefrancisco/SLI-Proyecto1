from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sl

class RegresionLineal:

    def __init__(self, dataset, target, trainSize = 0.8):
        self.datasetOriginal = dataset.copy()
        
        limiteParticion = int(len(dataset) * trainSize)
        self.datasetTrain = dataset[0 : limiteParticion, :]
        self.datasetTest = dataset[limiteParticion : , : ]
        self.target = target

    def AnalisisExploratorio(self):
        self.datasetOriginal.describe()

    def GenerarHistogramas(self):
        for columna in self.datasetOriginal.columns:
            sns.displot(self.datasetOriginal[columna])

    def AnalisisVariablesIndependientes(self):
        for columna in self.datasetTrain.columns:
            if columna != self.target:
                correlacion = self.datasetTrain[self.target].corr(self.datasetTrain[columna], method = "pearson")
                #plt.figure(figsize = (15,6))
                plt.xlabel(columna)
                plt.ylabel(self.target)
                plt.title(f"{columna} vs {self.target} - Correlación: {correlacion}")
                plt.scatter(self.datasetTrain[columna], self.datasetTrain[self.target])
                plt.show()
