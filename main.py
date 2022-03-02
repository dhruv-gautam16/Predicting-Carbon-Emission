#We used RandomForestRegressor for predicting the future carbon emission

# importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#reading the Data and Dropping the Columns

test=pd.read_csv('../input/ai-hackathon-foobar-60/testCO2.csv')
train=pd.read_csv('../input/ai-hackathon-foobar-60/trainCO2.csv')
train.drop('MODEL',inplace=True, axis=1)
test.drop('MODEL',inplace=True, axis=1)
train.drop('MAKE',inplace=True, axis=1)
test.drop('MAKE', inplace=True, axis=1)
train.drop('TRANSMISSION', inplace=True, axis=1)
test.drop('TRANSMISSION', inplace=True, axis=1)
train.drop('FUELTYPE', inplace=True, axis=1)
test.drop('FUELTYPE', inplace=True, axis=1)
train.drop('VEHICLECLASS', inplace=True, axis=1)
test.drop('VEHICLECLASS', inplace=True, axis=1)


#Making A Column to store the final value
train['Target CO2 Emission']=0
fid=train['Id']

#Converting Strings to Integer
l=LabelEncoder()
X=train.iloc[0:13].values
train.iloc[0:13]=l.fit_transform(X.astype('int64'))
X=train.iloc[0:799].values
train.iloc[0:799]=l.fit_transform(X.astype('int64'))


X=test.iloc[0:12].values
test.iloc[0:12]=l.fit_transform(X.astype('int64'))
X=test.iloc[0:264].values
test.iloc[0:264]=l.fit_transform(X.astype('int64'))

#Taking the Training Variable

y_train=train['Target CO2 Emission']
x_train=train.drop(['Target CO2 Emission'],axis=1)

#Taking Variables for training

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=0)

pip=Pipeline([('scaler2',StandardScaler()),('random_test:',RandomForestRegressor())])

#Training the Model
pip.fit(x_train,y_train)

prediction=pip.predict(x_test)

acc=pip.score(x_test,y_test)

acc

predict=pip.predict(test)
#Converting the output to dataframe


output=pd.DataFrame({'Id':fid,'Target CO2 Emission':predict})

output.head()