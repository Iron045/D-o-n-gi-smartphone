import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu và xử lý
@st.cache
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

# Huấn luyện mô hình
@st.cache
def train_model(model_type):
    X, y, cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Lasso Regression':
        model = Lasso(alpha=0.1)
    elif model_type == 'Neural Network':
        # Sử dụng pipeline với StandardScaler để chuẩn hóa dữ liệu trước khi huấn luyện mạng neural
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42))
    elif model_type == 'Stacking':
        # Stacking sử dụng LinearRegression làm mô hình chính và kết hợp Lasso làm mô hình con
        estimators = [
            ('lasso', Lasso(alpha=0.1))
        ]
        model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    
    model.fit(X_train, y_train)
    return model, cols

# Tạo input từ người dùng
def create_input_data(cols):
    # Chọn thương hiệu và hệ điều hành
    brand = st.selectbox('Chọn thương hiệu', ['Apple', 'Samsung', 'Xiaomi', 'Oppo', 'Realme'])
    os = st.selectbox('Chọn hệ điều hành', ['Android', 'iOS'])
    inches = st.slider('Kích thước màn hình (inches)', 4.0, 7.0, 6.0)
    width = st.slider('Độ phân giải (chiều rộng)', 600, 3000, 1080)
    height = st.slider('Độ phân giải (chiều cao)', 1000, 4000, 2400)
    battery = st.slider('Dung lượng pin (mAh)', 1000, 5000, 3000)
    ram = st.slider('Dung lượng RAM (GB)', 2, 12, 4)
    weight = st.slider('Trọng lượng (g)', 100, 300, 200)
    storage = st.slider('Dung lượng bộ nhớ trong (GB)', 16, 512, 128)

    # Tạo DataFrame cho đầu vào
    input_data = pd.DataFrame({
        'inches': [inches],
        'width': [width],
        'height': [height],
        'battery': [battery],
        'ram(GB)': [ram],
        'weight(g)': [weight],
        'storage(GB)': [storage]
    })
    
    # One-hot encoding cho thương hiệu và hệ điều hành
    brand_dict = {'brand_Apple': 0, 'brand_Samsung': 0, 'brand_Xiaomi': 0, 'brand_Oppo': 0, 'brand_Realme': 0}
    os_dict = {'os_Android': 0, 'os_iOS': 0}
    
    brand_dict[f'brand_{brand}'] = 1
    os_dict[f'os_{os}'] = 1
    
    for key, value in brand_dict.items():
        input_data[key] = value
    for key, value in os_dict.items():
        input_data[key] = value
    
    # Đảm bảo dữ liệu đầu vào có đủ các cột cần thiết
    for col in cols:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Sắp xếp cột theo thứ tự đã huấn luyện
    input_data = input_data[cols]

    return input_data

# Người dùng chọn mô hình
model_type = st.selectbox('Chọn mô hình dự đoán', ['Linear Regression', 'Lasso Regression', 'Neural Network', 'Stacking'])

# Tải và huấn luyện mô hình
model, cols = train_model(model_type)

# Tạo dữ liệu input từ người dùng
input_data = create_input_data(cols)

# Thêm nút dự đoán
if st.button("Dự đoán giá"):
    # Dự đoán giá
    try:
        predicted_price = model.predict(input_data)[0]
        # Hiển thị giá dự đoán
        st.title("Dự đoán giá Smartphone")
        st.subheader(f"Giá dự đoán: {predicted_price:.2f} USD")
    except ValueError as e:
        st.error(f"Đã xảy ra lỗi: {e}")
