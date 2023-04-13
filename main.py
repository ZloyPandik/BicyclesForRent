import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("bicycles for rent.xls")

df.head()
df.describe().T

linreg = LinearRegression(fit_intercept = True)

X = df.drop(["cnt", "atemp", "windspeed(mph)"], axis = 1)
y = df["cnt"]
df.info()

X.shape, y.shape



X_scal = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scal, y,
        test_size = 0.3, random_state=42)

linreg.fit(x_train, y_train)



y_pred = linreg.predict(x_test)



y_pred.shape

df["cnt"][100]

y_pred[100]

print(linreg.coef_,linreg.intercept_)