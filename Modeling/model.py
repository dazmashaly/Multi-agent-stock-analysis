import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("final_data_without_NV_PY.csv")
df.dropna(inplace=True)
error_columns = ["sarima_error", "autots_error", "moe_error"]
print(np.abs(df[error_columns]).mean())
df.drop(columns=["sarima_error", "autots_error", "moe_error", "date", "final_score"], inplace=True)


x = df.drop("label", axis=1)
y = df[["label"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

encoder = OrdinalEncoder()
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Support Vector Regression": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Neighbors": KNeighborsRegressor(),
    "Extra Trees": ExtraTreesRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} MAE: {mae}, R2: {r2}")
