import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

# Tải dữ liệu
@st.cache
def load_data():
    data = pd.read_csv('phone_prices.csv')
    data[['width', 'height']] = data['resolution'].str.split('x', expand=True)
    data['width'] = pd.to_numeric(data['width'])
    data['height'] = pd.to_numeric(data['height'])
    data_encoded = pd.get_dummies(data, columns=['brand', 'os'], drop_first=True)
    X = data_encoded.drop('price(USD)', axis=1)
    y = data_encoded['price(USD)']
    return X, y

# Tạo input từ người dùng
def user_input():
    brand = st.selectbox('Chọn thương hiệu', ['Apple', 'Samsung', 'Xiaomi', 'Oppo', 'Realme'])
    os = st.selectbox('Chọn hệ điều hành', ['Android', 'iOS'])
    inches = st.slider('Kích thước màn hình (inches)', 4.0, 7.0, 6.0)
    width = st.slider('Độ phân giải (chiều rộng)', 600, 3000, 1080)
    height = st.slider('Độ phân giải (chiều cao)', 1000, 4000, 2400)
    battery = st.slider('Dung lượng pin (mAh)', 1000, 5000, 3000)
    ram = st.slider('Dung lượng RAM (GB)', 2, 12, 4)
    weight = st.slider('Trọng lượng (g)', 100, 300, 200)
    storage = st.slider('Dung lượng bộ nhớ trong (GB)', 16, 512, 128)
    
    input_data = pd.DataFrame({
        'inches': [inches],
        'width': [width],
        'height': [height],
        'battery': [battery],
        'ram(GB)': [ram],
        'weight(g)': [weight],
        'storage(GB)': [storage],
        f'brand_{brand}': [1],
        f'os_{os}': [1]
    })
    
    return input_data

# Hiển thị giao diện người dùng và tạo biểu đồ
def plot_error_distributions(errors, models):
    st.title("Đồ thị sai số")
    fig, axs = plt.subplots(1, len(models), figsize=(20, 5))
    fig.suptitle('Phân phối sai số', fontsize=20)
    
    for i, (model, error) in enumerate(errors.items()):
        axs[i].hist(error, bins=50)
        axs[i].set_title(f"{model} - Phân phối sai số")
        axs[i].set_xlabel('Error')
        axs[i].set_ylabel('Frequency')
    
    st.pyplot(fig)

def plot_actual_vs_pred(predictions, y_test, models):
    st.title("So sánh giá trị thực và dự đoán")
    fig, axs = plt.subplots(1, len(models), figsize=(20, 5))
    fig.suptitle('So sánh giá trị thực và dự đoán', fontsize=20)
    
    for i, (model, y_pred) in enumerate(predictions.items()):
        axs[i].scatter(y_test, y_pred, alpha=0.5)
        axs[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        axs[i].set_title(f"{model} - So sánh giá trị")
        axs[i].set_xlabel('Giá trị thực tế')
        axs[i].set_ylabel('Giá trị dự đoán')
    
    st.pyplot(fig)

# Huấn luyện và dự đoán
def run_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "MLP": MLPRegressor(max_iter=1000),
        "Stacking": StackingRegressor(estimators=[
            ('lr', LinearRegression()), 
            ('lasso', Lasso()), 
            ('mlp', MLPRegressor(max_iter=1000))])
    }

    predictions = {}
    errors = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        errors[name] = y_test - y_pred  # Sai số
    
    return predictions, errors

# Chạy ứng dụng
def main():
    st.title('Dự đoán giá smartphone')
    
    # Tải dữ liệu
    X, y = load_data()

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Lấy dữ liệu đầu vào từ người dùng
    input_data = user_input()

    # Huấn luyện mô hình và tạo dự đoán
    predictions, errors = run_models(X_train, y_train, X_test, y_test)

    # Hiển thị biểu đồ sai số và biểu đồ so sánh giá trị thực
    plot_error_distributions(errors, predictions)
    plot_actual_vs_pred(predictions, y_test, predictions)

    # Dự đoán cho dữ liệu người dùng
    model_choice = st.selectbox('Chọn mô hình dự đoán', ['Linear Regression', 'Lasso', 'MLP', 'Stacking'])
    model = models[model_choice]
    user_pred = model.predict(input_data)
    st.subheader(f'Giá dự đoán cho smartphone của bạn là: {user_pred[0]:.2f} USD')

if __name__ == '__main__':
    main()
