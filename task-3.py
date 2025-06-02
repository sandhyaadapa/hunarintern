import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# step 1:Load Data
df=pd.read_csv("breast cancer.csv")
# Drop the 'id' column
df.drop("id",axis=1,inplace=True)
# Encode target:M->1,B->0
df['diagnosis']=df['diagnosis'].map({'M':1, 'B':0})
# step 2:Data preprocessing
X=df.drop("diagnosis",axis=1)
y=df["diagnosis"]
# Normalize features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
# split into train-test
X_train, X_test, y_train, y_test=train_test_split(X_scaled, y)
# step 3:Apply K-NN Algorithm
k=7 #Default choice
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
# step 4:Model Evaluation
y_pred=knn.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

