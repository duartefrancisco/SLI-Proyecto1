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

    def ObtenerColumna(self, columna, dataset = "train"):
        if dataset.lower() == "train":
            return self.datasetTrain[columna]
        elif dataset.lower() == "test":
            return self.datasetTest[columna]
        elif dataset.lower() == "original":
            return self.datasetOriginal[columna]

    def Entrenar(self, variableDependiente, variableIndependiente, epochs, imprmir_error_cada, learningRate):
        unos = np.ones(np.shape(variableDependiente)).reshape(-1,1)
        x = variableDependiente.to_numpy().reshape(-1,1)
        x = np.hstack([x, unos])
        y = variableIndependiente.to_numpy()
        beta0 = 0
        beta1 = 0
        errores = []
        modelos = {}

        for i in range(epochs):
            modelos[i] = [beta0, beta1]
            betas = np.array([beta1, beta0]).reshape(-1, 1)
            yEstimado = np.matmul(x, betas)
            gradienteB0 = np.mean(yEstimado - y)
            gradienteB1 = np.mean((yEstimado - y) * x)
            beta0 = beta0 - learningRate * gradienteB0
            beta1 = beta1 - learningRate * gradienteB1

            if (i % imprmir_error_cada) == 0:
                errores.append(np.mean((yEstimado - y)**2) * 1/2)
        
        return modelos, errores


