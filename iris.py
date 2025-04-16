import pandas as pd 
import numpy as np 
import pickle 

from sklearn.metrics import accuracy_score
df=pd.read_csv('iris.data.csv',header=None)
x=np.array(df.iloc[:,0:4])
y=np.array(df.iloc[:,4:])
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
y=le.fit_transform(y) 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2) 
from sklearn.svm import SVC 
sv=SVC(kernel='linear').fit(X_train,Y_train)
pickle.dump(sv,open('iris.pkl','wb'))
