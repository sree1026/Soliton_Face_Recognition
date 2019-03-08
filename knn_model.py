import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

filename = input("Enter CSV Filename: ")
# read the csv file containing the encodings
df = pd.read_csv(filename, header=None)
# separate the encodings from the csv file
encodings = df.drop(df.columns[0], axis=1)
# separate the class name i.e name of person from the csv file
names = df[0]
# specify number of neighbours for tthe model
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model
knn.fit(encodings, names)
filename = 'knn_classifier_model.sav'
# Store the model for later use
joblib.dump(knn, filename)
print("KNN Model trained and stored....")
