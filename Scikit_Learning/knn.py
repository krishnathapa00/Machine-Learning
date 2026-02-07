# K-NN Classificaion Model:
# fruits prediction based on weight and size.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#Data
x=[
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [330,9],
    [360,9.5],
    ]

#0=apple, 1=orange
y=[0,0,0,1,1,1] 
#Model
model=KNeighborsClassifier(n_neighbors=3) #n_neighbors is the number of nearest neighbors to use for prediction.
model.fit(x,y) #fitting the model to the data

weight=float(input("Enter the weight of the fruit in grams: "))
size=float(input("Enter the size of the fruit in cms: "))

#Predicting
fruit=model.predict([[weight,size]]) [0]       #predict method is used to predict the class label for the given input data.
if fruit==0:    #if the predicted class label is 0, it means the fruit is an apple.
    print("The fruit is an apple.")
else:    print("The fruit is an orange.") #if the predicted class label is 1, it means the fruit is an orange.
