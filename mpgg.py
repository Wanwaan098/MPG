import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler  
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv('auto-mpg.csv') 
print(df.head())
df.shape


missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)


rows_to_drop = df.isin(['?', '', 'None', 'NaN']).any(axis=1)
df_with_missing = df[rows_to_drop]
print(df_with_missing)


df.drop(df_with_missing.index, inplace=True)

missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)


df['horsepower'] = df['horsepower'].astype(int)
df.info()


df['origin'] = df['origin'].map({1: 'US', 2: 'Asia', 3: 'Europe'})
df['origin'].value_counts()


df['model year'] = (1900 + df['model year'])
print(df.head())



df_numeric = df.select_dtypes(include=[np.number])
df_corr = df_numeric.corr()
sns.heatmap(df_corr, annot=True, cmap="RdYlBu", fmt=".2f", linewidths=0.5)
plt.show()


df_numeric = df.select_dtypes(include=[np.number])
df_corr = df_numeric.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
df_corr.rename(columns={"level_0": "Feature A", 
                             "level_1": "Feature B", 0: 'Correlation Coefficient'}, inplace=True)
df_corr[df_corr['Feature A'] == 'mpg']
print(df_corr[df_corr['Feature A'] == 'mpg'])


df.drop(["acceleration", "displacement","car name"], axis = 1, inplace = True)
print(df.head())


df = pd.get_dummies(df, drop_first = True)
print(df.head())


X = df.drop(columns=["mpg"]) 
y = df["mpg"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
mean_mae_linear = -cross_val_score(estimator=linear_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_absolute_error').mean()
mean_mse_linear = -cross_val_score(estimator=linear_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error').mean()
mean_rmse_linear = np.sqrt(mean_mse_linear)
mean_r2_linear = cross_val_score(estimator=linear_model, X=X_train, y=y_train, cv=10, scoring='r2').mean()


ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
mean_mae_ridge = -cross_val_score(estimator=ridge_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_absolute_error').mean()
mean_mse_ridge = -cross_val_score(estimator=ridge_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error').mean()
mean_rmse_ridge = np.sqrt(mean_mse_ridge)
mean_r2_ridge = cross_val_score(estimator=ridge_model, X=X_train, y=y_train, cv=10, scoring='r2').mean()


nn_model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), random_state=42, max_iter=1000, 
                        activation='relu', solver='adam', learning_rate_init=0.001)
nn_model.fit(X_train_scaled, y_train)
mean_mae_nn = -cross_val_score(estimator=nn_model, X=X_train_scaled, y=y_train, cv=10, scoring='neg_mean_absolute_error').mean()
mean_mse_nn = -cross_val_score(estimator=nn_model, X=X_train_scaled, y=y_train, cv=10, scoring='neg_mean_squared_error').mean()
mean_rmse_nn = np.sqrt(mean_mse_nn)
mean_r2_nn = cross_val_score(estimator=nn_model, X=X_train_scaled, y=y_train, cv=10, scoring='r2').mean()


estimators = [('linear', linear_model), ('ridge', ridge_model), ('neural_net', nn_model)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)
mean_mae_stack = -cross_val_score(estimator=stacking_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_absolute_error').mean()
mean_mse_stack = -cross_val_score(estimator=stacking_model, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error').mean()
mean_rmse_stack = np.sqrt(mean_mse_stack)
mean_r2_stack = cross_val_score(estimator=stacking_model, X=X_train, y=y_train, cv=10, scoring='r2').mean()


performance = pd.DataFrame({
    'Metrics': ['MAE', 'MSE', 'RMSE', 'Score'],
    'Linear Regression': [
        f"{mean_mae_linear:.4f}",
        f"{mean_mse_linear:.4f}",
        f"{mean_rmse_linear:.4f}",
        f"{mean_r2_linear * 100:.2f}%"
    ],
    'Ridge Regression': [
        f"{mean_mae_ridge:.4f}",
        f"{mean_mse_ridge:.4f}",
        f"{mean_rmse_ridge:.4f}",
        f"{mean_r2_ridge * 100:.2f}%"
    ],
    'Neural Network': [
        f"{mean_mae_nn:.4f}",
        f"{mean_mse_nn:.4f}",
        f"{mean_rmse_nn:.4f}",
        f"{mean_r2_nn * 100:.2f}%"
    ],
    'Stacking Model': [
        f"{mean_mae_stack:.4f}",
        f"{mean_mse_stack:.4f}",
        f"{mean_rmse_stack:.4f}",
        f"{mean_r2_stack * 100:.2f}%"
    ]
})


print(performance.set_index('Metrics'))
