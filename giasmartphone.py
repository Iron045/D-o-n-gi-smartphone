import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tải dữ liệu và xử lý
@st.cache
def load_data():
    data = pd.read_csv('phone_prices.csv')
    data[['width', 'height']] = data['resolution'].str.split('x', expand=True)
    data['width'] = pd.to_numeric(data['width'])
    data['height'] = pd.to_numeric(data['height'])
    
    data_filtered = data[['brand', 'os', 'inches', 'width', 'height', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'price(USD)']]
    data_encoded = pd.get_dummies(data_filtered, columns=['brand', 'os'], drop_first=True)
    X = data_encoded.drop('price(USD)', axis=1)
    y = data_encoded['price(USD)']
    
    return X, y, data_encoded.columns

# Huấn luyện mô hình
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "MLP": MLPRegressor(max_iter=1000),
        "Stacking": StackingRegressor(estimators=[
            ('lr', LinearRegression()), 
            ('lasso', Lasso()), 
            ('mlp', MLPRegressor(max_iter=1000))])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

# Đánh giá mô hình
def evaluate_models(models, X_test, y_test):
    evaluation_results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'R²': r2
        })

    return pd.DataFrame(evaluation_results)

# Tạo input từ người dùng
def create_input_data(cols):
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
    
    # Đảm bảo rằng dữ liệu đầu vào có đầy đủ các cột như trong dữ liệu huấn luyện
    for col in cols:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Sắp xếp cột theo đúng thứ tự
    input_data = input_data[cols]
    
    return input_data

# Chạy ứng dụng
def main():
    st.title('Dự đoán giá smartphone')

    # Tải dữ liệu
    X, y, cols = load_data()

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện các mô hình
    models = train_models(X_train, y_train)

    # Đánh giá mô hình
    evaluation_df = evaluate_models(models, X_test, y_test)
    
    # Hiển thị bảng đánh giá mô hình
    st.subheader("Đánh giá mô hình")
    st.dataframe(evaluation_df)

    # Tạo dữ liệu input từ người dùng
    input_data = create_input_data(cols)

    # Dự đoán giá cho dữ liệu người dùng
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_data)[0]

    # Hiển thị kết quả dự đoán
    st.subheader("Kết quả dự đoán")
    results_df = pd.DataFrame({
        'Model': predictions.keys(),
        'Dự đoán': predictions.values()
    })
    st.dataframe(results_df)

if __name__ == '__main__':
    main()
