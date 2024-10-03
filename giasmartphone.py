import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    X = data_encoded.drop('price(USD)', axis=1)
    y = data_encoded['price(USD)']
    
    return X, y, data_encoded.columns

# Huấn luyện mô hình
@st.cache
def train_model():
    X, y, cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
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
    brand_dict = {'Apple': 0, 'Samsung': 0, 'Xiaomi': 0, 'Oppo': 0, 'Realme': 0}
    os_dict = {'Android': 0, 'iOS': 0}
    
    brand_dict[brand] = 1
    os_dict[os] = 1
    
    for key, value in brand_dict.items():
        input_data[f'brand_{key}'] = value
    for key, value in os_dict.items():
        input_data[f'os_{key}'] = value
    
    # Đảm bảo dữ liệu đầu vào có đủ các cột cần thiết
    for col in cols:
        if col not in input_data.columns:
            input_data[col] = 0
    
    return input_data

# Tải và huấn luyện mô hình
model, cols = train_model()

# Tạo dữ liệu input từ người dùng
input_data = create_input_data(cols)

# Dự đoán giá
try:
    predicted_price = model.predict(input_data)[0]
    # Hiển thị giá dự đoán
    st.title("Dự đoán giá Smartphone")
    st.subheader(f"Giá dự đoán: {predicted_price:.2f} USD")
except ValueError as e:
    st.error(f"Đã xảy ra lỗi: {e}")
