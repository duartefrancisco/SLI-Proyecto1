from this import s
from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

class RegresionLineal:

    def __init__(self, dataset, target, trainSize = 0.8):
        self.datasetOriginal = dataset.copy()
        
        limiteParticion = int(len(dataset) * trainSize)
        self.datasetTrain = dataset.iloc[0 : limiteParticion, :]
        self.datasetTest = dataset.iloc[limiteParticion : , : ]
        self.target = target

    def AnalisisExploratorio(self):
        print(self.datasetOriginal.describe())

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

    
    def __CalcularPrediccion(self, x, betas):
        tempBetas = np.array(betas).reshape(-1, 1)
        
        return np.matmul(x, tempBetas)

    def Entrenar(self, variableDependiente, variableIndependiente, epochs, imprmir_error_cada, learningRate):
        unos = np.ones(np.shape(variableDependiente)).reshape(-1,1)
        X = variableIndependiente.to_numpy().reshape(-1,1)
        A = np.hstack([X, unos])
        y = variableDependiente.to_numpy()
        beta0 = 0
        beta1 = 0
        errores = []
        modelos = {}
        
        for i in range(epochs):
            #modelos = pd.concat([modelos, pd.DataFrame([i, [beta0, beta1]], columns= ["Intento", "Betas"])], ignore_index= True)
            modelos[i] = [beta1, beta0]
            yEstimado = self.__CalcularPrediccion(A, [beta1, beta0])
            gradienteB0 = np.mean(yEstimado - y)
            gradienteB1 = np.mean((yEstimado - y) * X)
            beta0 = beta0 - learningRate * gradienteB0
            beta1 = beta1 - learningRate * gradienteB1
            error = np.mean((yEstimado - y)**2) * 1/2
            errores.append(error)
            if (i % imprmir_error_cada) == 0:
                print(f"Iteración = {i}, Error = {error}")
                
        return modelos, errores

    def VisualizacionError(self, errores):
        #np.array(range(1,len(errores)+1))
        plt.scatter(np.array(range(1,len(errores)+1)), np.array(errores))
        plt.show()
    
    def VisualizacionDelModelo(self, modelos, n, variable):
        #tempX = random.sample(range(200), 15).to_numpy().shape(-1,1)
        tempX = self.datasetTrain[variable].to_numpy().reshape(-1,1)
        tempBetas = modelos.copy()
        #tempBetas = tempBetas.to_numpy()
        unos = np.ones(np.shape(self.datasetTrain[variable])).reshape(-1,1)
        x = np.hstack([tempX, unos])

        fig = plt.figure()   
        ax = fig.add_subplot(111)
        for i in range(len(tempBetas)):
            if (i % n) == 0:
                y = self.__CalcularPrediccion(x, tempBetas[i])

                ax.plot(tempX, y)
                
        plt.xlabel("Datos Entrenamiento")
        plt.ylabel("Estimación")
        plt.show()

        print("Datos entrenamiento\n")
        print(pd.DataFrame(tempX))
        print("Fin Datos entrenamiento")

    def SeleccionarModelo(self, modelos, errores):
        errorMinimo = np.argmin(errores)
        
        return modelos[errorMinimo]
        

    def EntrenarSklearn(self, variable):
        tempX = self.datasetTrain[variable].to_numpy().reshape(-1,1)
        tempY = self.datasetTrain[self.target].to_numpy().reshape(-1,1)

        reg = LinearRegression()
        reg.fit(tempX, tempY)

        return reg

    def Predicciones(self, modeloMnaual, modeloSklearn, x):
        tempX = np.array(x).reshape(-1,1)
        prediccionPromedio = []
        unos = np.ones(np.shape(x)).reshape(-1,1)
        betasManual = np.array(modeloMnaual).reshape(-1,1)
        X = np.hstack([tempX, unos])
        
        yManual = self.__CalcularPrediccion(X, modeloMnaual)
        ySklearn = modeloSklearn.predict(tempX)
        prediccionPromedio = (yManual + ySklearn)/2
            
        return yManual, ySklearn, prediccionPromedio


    def PrediccionesTest(self, variable, modelo):
        tempX = self.datasetTest[variable].to_numpy().reshape(-1,1)
        y = self.datasetTest[self.target].to_numpy()
        betas = np.array(modelo).reshape(-1,1)
        unos = unos = np.ones(np.shape(self.datasetTest[variable])).reshape(-1,1)
        x = np.hstack([tempX, unos])
        

        yEstimado = self.__CalcularPrediccion(x, betas)
        errores = yEstimado[0] - y

        print(modelo)
        plt.scatter(self.datasetTest[variable], self.datasetTest[self.target])
        plt.plot(self.datasetTest[variable], yEstimado, color = "black")
        plt.title(f"Regresión Lineal Simple - {variable} -Data Train")
        plt.show()

        return errores

    def PrediccionesTestSklearn(self, variable, modelo):
        tempX = self.datasetTest[variable].to_numpy().reshape(-1,1)
        y = self.datasetTest[self.target].to_numpy()        

        yEstimado = modelo.predict(tempX)
        errores = yEstimado[0] - y
        
        plt.scatter(self.datasetTest[variable], self.datasetTest[self.target])
        plt.plot(self.datasetTest[variable], yEstimado, color = "black")
        plt.title(f"Regresión Lineal Simple - {variable} -Data Train")
        plt.show()

        return errores

    def VisualizacionErrorTest(self, variable, errores):
        xRango = range(1,len(self.datasetTest[variable])+1)
        error = np.mean((errores)**2) * 1/2
        plt.scatter(xRango, errores)
        plt.title(f"Erroes del modelo para {variable} - Error: {error}")
        plt.show() 



        

