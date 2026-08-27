import pandas as pd #used for functions to analyze data 
from sklearn.ensemble import RandomForestClassifier #Import randomfc from sklearn, randomfc is a ML model that builds decision trees on random subsets of data and combines their results
from sklearn.model_selection import train_test_split #Splits arrays and matrices into random train and test subsets
import pickle #In this case pickle is used to turne the ML model into a binary byte stream so it can be saved into a file

#Read data from the csv file using pandas function
data = pd.read_csv('hand_coords', header=None)

coords_x = data.iloc[:, :-1] #Choose all the cords
coords_y = data.iloc[:, -1] #Choose all the lables to the coresponding coords

model = RandomForestClassifier() #Create the model used to recognise what signs is used
model.fit(coords_x, coords_y) #Tells the model what data from the csv it should use

with open('model.p', 'wb') as f:
    pickle.dump(model, f) #Turns the model into a byte stream and saves it into a file.

print("Model training done and saved as model.p")