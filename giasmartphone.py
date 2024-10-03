# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/phone_prices.csv')  # Thay bằng đường dẫn chính xác tới file dữ liệu của bạn

# Hiển thị 5 dòng đầu tiên của tập dữ liệu để kiểm tra
print(data.head())

# Tách cột resolution thành hai cột riêng biệt: width và height
data[['width', 'height']] = data['resolution'].str.split('x', expand=True)

# Chuyển đổi cột width và height sang kiểu số
data['width'] = pd.to_numeric(data['width'])
data['height'] = pd.to_numeric(data['height'])

# Lọc các cột không cần thiết, bỏ các cột liên quan đến video và các cột không liên quan khác
data_filtered = data[['brand', 'os', 'inches', 'width', 'height', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'price(USD)']]

# Xử lý các cột phân loại (brand, os) bằng One-Hot Encoding
data_encoded = pd.get_dummies(data_filtered, columns=['brand', 'os'], drop_first=True)

# Xác định biến đầu vào (X) và biến mục tiêu (y)
X = data_encoded.drop('price(USD)', axis=1)
y = data_encoded['price(USD)']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình hồi quy tuyến tính
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Mô hình hồi quy Ridge
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)

# Mô hình mạng nơ-ron đơn giản (MLP Regressor)
model_nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)

# Mô hình Bagging Regressor
model_bagging = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
model_bagging.fit(X_train, y_train)
y_pred_bagging = model_bagging.predict(X_test)

# Mô hình Stacking Regressor
estimators = [('lr', LinearRegression()), ('ridge', Ridge(alpha=1.0))]
model_stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
model_stacking.fit(X_train, y_train)
y_pred_stacking = model_stacking.predict(X_test)

# Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R²: {r2}")
    return mse, r2

# Đánh giá tất cả các mô hình
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_nn, "Neural Network")
evaluate_model(y_test, y_pred_bagging, "Bagging Regressor")
evaluate_model(y_test, y_pred_stacking, "Stacking Regressor")


# Giả sử bạn đã huấn luyện mô hình và có các dữ liệu sau:
# y_train thực, y_train dự đoán, y_test thực, y_test dự đoán, y_val thực, y_val dự đoán

# Giá trị dự đoán của tập huấn luyện và kiểm tra
y_train_pred = model_lr.predict(X_train)
y_test_pred = model_lr.predict(X_test)


# Hiển thị đồ thị so sánh giá trị thực tế và giá trị dự đoán của một mô hình (Linear Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Predicted Price')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression - Actual vs Predicted Prices')
plt.legend()
plt.show()
