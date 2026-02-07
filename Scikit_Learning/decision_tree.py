# Decision Tree Classifier-a model that makes predictions based on a series of yes/no questions.

# Example - fruits prediction based on size(cms) & shades(1-10).

from sklearn.tree import DecisionTreeClassifier
#Data
x=[[7,2],
   [8,3],
   [6,1],
   [5,2],
   [9,4],
]
#0=apple, 1=orange
y=[0,0,0,1,1]
#Model
model=DecisionTreeClassifier() #creating an instance of the DecisionTreeClassifier class.
model.fit(x,y) #fitting the model to the data   

size=float(input("Enter the size of the fruit in cms: "))
shade=float(input("Enter the shade of the fruit (1-10): "))
#Predicting
result=model.predict([[size,shade]]) [0]       #predict method is used to predict the class label for the given input data.
if result==0:    #if the predicted class label is 0, it means the fruit is an apple.
    print("The fruit is an apple.")
else:    print("The fruit is an orange.") #if the predicted class label is 1, it means the fruit is an orange.