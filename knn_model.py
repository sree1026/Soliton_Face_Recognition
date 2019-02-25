import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import time

start = time.time()
df = pd.read_csv('output4.csv', header=None)
print("Excel loading time::: "+str(time.time()-start))

df.head()
start = time.time()
# print(start)
X = df.drop(df.columns[0], axis=1)

y = df[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=False)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)
print("Model training time :: "+str(time.time()-start))
filename = 'knn_classifier_model.sav'

joblib.dump(knn, filename)
