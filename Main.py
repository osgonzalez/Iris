# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
# numpy
import numpy as np
# Iris Dataset
from sklearn import datasets
irisDataset = datasets.load_iris()
# Dataset Shuffle and Split Funtion
from sklearn.model_selection import train_test_split
# Preprocesing Functions
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from numpy import argmax

# OS Untils
import os.path

# Models and Layers
from keras.models import Sequential
from keras.layers.core import Dense

#HiperParameters
EPOCHS = 1000

#Transform <class 'sklearn.utils.Bunch'>  to  <class 'pandas.core.frame.DataFrame'>
irisDataframe = pd.DataFrame(irisDataset["data"],columns=irisDataset["feature_names"])

#Scale the imput data
#scalar = MinMaxScaler()
#scalar.fit(irisDataframe)
#irisDataframe = scalar.transform(irisDataframe)

#Encoding the target Data in "One Hot Encoding"
targetOneHot = to_categorical(pd.Series(irisDataset["target"]))

# Dataset Shuffle and Split
vars_train, vars_test, target_train, target_test = train_test_split(irisDataframe, targetOneHot )


#Create the model
model = Sequential() # Model Secuential Factory (Este modelo ordena las capas de forma secuencial)
model.add(Dense(8, activation='relu', input_dim=4)) #Fist Hide layer (and imput_dim =4, define the imput layer with 4 imputs)
model.add(Dense(3, activation='sigmoid'))   #Dense = Hidden Layer

#Compile the model, and get the Loss and Optimization Functions
model.compile(loss='mean_squared_error', optimizer='sgd')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#lossFuntion = losses.mean_squared_error
#model.compile(loss=lossFuntion, optimizer=sgd)


print(model.summary())


#Fit the model whith the train data (and indicate the numer of iterations for the data)
model.fit(vars_train, target_train, epochs=EPOCHS)


scores = model.evaluate(vars_train, target_train)
print("\nScores of Train Data %s: %.2f%%" % (model.metrics_names, scores*100))

scores = model.evaluate(vars_test, target_test)
print("\nScores of Test Data %s: %.2f%%" % (model.metrics_names, scores*100))

print("\n",model.predict(np.array([[5.1,3.5,1.4,0.2]], "float32"))) # Setosa
print(model.predict(np.array([[5.7,3.8,1.7,0.3]], "float32"))) # Setosa

print("\n",model.predict(np.array([[7.0,3.2,4.7,1.4]], "float32"))) # versicolor
print(model.predict(np.array([[6.0,2.2,4.0,1.0]], "float32"))) # versicolor

print("\n",model.predict(np.array([[5.1,3.5,1.4,0.2]], "float32"))) # virginica
print(model.predict(np.array([[6.0,2.2,5.0,1.5]], "float32"))) # virginica

print("\n\n")



correctPredictions = 0
failurePredictions = 0

for i in range(target_test.shape[0]): #The shape atribute is a tuple with the number of elements of each dimension
    dataRow = np.array([[vars_test.iloc[i][0],vars_test.iloc[i][1],vars_test.iloc[i][2],vars_test.iloc[i][3]]], "float32")
    targetRow = argmax(target_test[i]) #The argmax function return the position of the max value
    prediction = model.predict(dataRow)
    print("\nThe espected results for the imputs:", dataRow, "was", targetRow) 
    #print("The prediction is:", model.predict(row), "-", model.predict_classes(row)  )
    print("The prediction is:", prediction , "-", argmax(prediction))
    if targetRow == argmax(prediction):
        print("Correct")
        correctPredictions += 1
    else:
        print("False")
        failurePredictions += 1


print("------- Results -------")
print("Sucess Predictions:", correctPredictions,"/",target_test.shape[0], "(",(correctPredictions/target_test.shape[0])*100,"% )")
print("Failure Predictions:", failurePredictions,"/",target_test.shape[0], "(",(failurePredictions/target_test.shape[0])*100,"% )")

if not os.path.exists('StatsFile'):
    statsFile = open("StatsFile","a")
    line = "Epochs \tCorrect"
    statsFile.write(line)
    statsFile.close()

statsFile = open("StatsFile","a")
line = "\n"+ str(EPOCHS) + "\t" + str((correctPredictions/target_test.shape[0])*100)
statsFile.write(line)
statsFile.close()
