import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('auto-mpg.csv') 
print(df.head())
df.shape



sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

numeric_cols = df.select_dtypes(include=[np.number]).columns
for i, feature in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True, color='blue', bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()



missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)


rows_to_drop = df.isin(['?', '', 'None', 'NaN']).any(axis=1)
df_with_missing = df[rows_to_drop]
print(df_with_missing)


df.drop(df_with_missing.index, inplace=True)

missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)


df['horsepower'] = df['horsepower'].astype(int)

df['origin'] = df['origin'].map({1: 'US', 2: 'Asia', 3: 'Europe'})
df['origin'].value_counts()
print(df.head())

df['model year'] = (1900 + df['model year'])
print(df.head())


df.drop(["acceleration", "displacement","car name"], axis = 1, inplace = True)
print(df.head())


df = pd.get_dummies(df, drop_first = True)
print(df.head())


X = df.drop(columns=["mpg"]) 
y = df["mpg"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Định nghĩa các mô hình
linear_model = LinearRegression()
ridge_model = Ridge(alpha=0.1)
nn_model = MLPRegressor(random_state=42, max_iter=2000, learning_rate_init=0.001, hidden_layer_sizes=(100, 50))

# Áp dụng Stacking
estimators = [('linear', linear_model), ('ridge', ridge_model), ('neural_net', nn_model)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# Hàm tính hiệu suất
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    rmse = np.sqrt(mse)
    r2 = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
    return mae, mse, rmse, r2 * 100

# Huấn luyện và đánh giá mô hình
models = [("Linear Regression", linear_model, X_train),
          ("Ridge Regression", ridge_model, X_train),
          ("Neural Network", nn_model, X_train_scaled),
          ("Stacking Model", stacking_model, X_train)]

performance_data = []

for name, model, X_data in models:
    model.fit(X_data, y_train)
    mae, mse, rmse, r2 = evaluate_model(model, X_data, y_train)
    performance_data.append([name, f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.2f}%"])

# Tạo DataFrame kết quả
performance = pd.DataFrame(performance_data, columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score']).set_index('Model')
print(performance)
