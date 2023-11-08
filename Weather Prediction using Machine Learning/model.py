import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pickle

df = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\Class Project\House-price-prediction-using-flask-main\House-price-prediction-using-flask-main\WF-dataset - Sheet1 (1).csv',encoding='latin1')

columns = ['State', 'Month', 'Day', 'Year', 'C']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



"""lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score=lr.score(X_test, y_test)
print(lr_score)"""

lr_lasso = Lasso()
lr_lasso.fit(X_train, y_train)
lr_lasso_score=lr_lasso.score(X_test, y_test)
#print(lr_lasso_score)



pickle.dump(lr_lasso, open('weather.pkl', 'wb'))
