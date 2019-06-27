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


# Models and Layers
from keras.models import Sequential
from keras.layers.core import Dense



#Transform <class 'sklearn.utils.Bunch'>  to  <class 'pandas.core.frame.DataFrame'>
irisDataframe = pd.DataFrame(irisDataset["data"],columns=irisDataset["feature_names"])

#Encoding the target Data in "One Hot Encoding"
targetOneHot = to_categorical(pd.Series(irisDataset["target"]))

# Dataset Shuffle and Split
vars_train, vars_test, target_train, target_test = train_test_split(irisDataframe, targetOneHot )


print(vars_train)
print(target_train)


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
model.fit(vars_train, target_train, epochs=100)


scores = model.evaluate(vars_train, target_train)
print("\nScores of Train Data %s: %.2f%%" % (model.metrics_names, scores*100))

scores = model.evaluate(vars_test, target_test)
print("\nScores of Test Data %s: %.2f%%" % (model.metrics_names, scores*100))

print(model.predict(np.array([[5.1,3.5,1.4,0.2]], "float32")))


