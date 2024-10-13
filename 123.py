from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    # Dự đoán trên tập kiểm tra và tính toán các tham số đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, cols, mae, mse, r2
