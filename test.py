import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv("diabetes.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting the logistic regression to the training set
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(max_iter = 1000)
classifier.fit(X_train, y_train) #putting our both train set into logistic classifier object so that it can learn the co realtion btw them 

#Predicting the test set result 
y_pred = classifier.predict(X_test)
#check y_pred and y_test for clear results, cz y pred is the prediction of x_test whose 
#actuall results are in y_test

#Making the confusion Matrix 
from sklearn.metrics import confusion_matrix # it's a function 
cm = confusion_matrix(y_test, y_pred  ) #it will tally the predicted results and ground truth results and show the accuray 


#Ploting the Confusion Matrics
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

import pickle
import os
#Saving the model
if not os.path.exists('models'):
    os.makedirs('models')
    
MODEL_PATH = "models/logistic_reg.sav"
#pickle.dump(logreg, open(MODEL_PATH, 'rb'))
pickle.load(MODEL_PATH)

# Enter your data here.
data = [[8,0,23.3,32,183,64,0.672]]
  
# Create the pandas DataFrame 
new_data = pd.DataFrame(data, columns = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']) 

#Predict On new Data
new_pred = classifier.predict(new_data)
print(new_pred)

