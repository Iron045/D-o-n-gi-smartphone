import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Tải dữ liệu và xử lý
def load_data():
    data = pd.read_csv('phone_prices.csv')  # Đường dẫn file CSV của bạn
    # Tách cột resolution thành width và height
    data[['width', 'height']] = data['resolution'].str.split('x', expand=True)
    data['width'] = pd.to_numeric(data['width'])
    data['height'] = pd.to_numeric(data['height'])
    
    # Lọc các cột liên quan
    data_filtered = data[['brand', 'os', 'inches', 'width', 'height', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'price(USD)']]
    
    # One-hot encoding cho các cột phân loại
    data_encoded = pd.get_dummies(data_filtered, columns=['brand', 'os'], drop_first=True)
    X = data_encoded.drop('price(USD)', axis=1)  # Loại bỏ cột mục tiêu 'price(USD)'
    y = data_encoded['price(USD)']
    
    return X, y, X.columns

# Tối ưu hóa Lasso Regression với GridSearchCV
def optimize_lasso(X_train, y_train):
    lasso = Lasso()

    # Định nghĩa tham số cần tối ưu
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Sử dụng GridSearchCV để tìm tham số tốt nhất cho Lasso
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("Best parameters for Lasso:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model

# Tối ưu hóa Neural Network với GridSearchCV
def optimize_neural_network(X_train, y_train):
    mlp = MLPRegressor(random_state=42)

    # Định nghĩa tham số cần tối ưu cho MLPRegressor
    param_grid = {
        'hidden_layer_sizes': [(64,), (64, 64), (128, 64)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000]
    }

    # Sử dụng pipeline với StandardScaler để chuẩn hóa dữ liệu
    pipeline = make_pipeline(StandardScaler(), mlp)

    # Sử dụng GridSearchCV để tìm tham số tốt nhất cho Neural Network
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("Best parameters for Neural Network:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model

# Huấn luyện mô hình
def train_and_evaluate_models():
    # Tải dữ liệu
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tối ưu hóa Lasso Regression
    lasso_model = optimize_lasso(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    print(f"Lasso RMSE: {lasso_rmse:.2f}")

    # Tối ưu hóa Neural Network
    nn_model = optimize_neural_network(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    nn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    print(f"Neural Network RMSE: {nn_rmse:.2f}")

# Gọi hàm để huấn luyện và tối ưu mô hình
if __name__ == "__main__":
    train_and_evaluate_models()
