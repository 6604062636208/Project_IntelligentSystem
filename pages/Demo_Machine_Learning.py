import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import io
import pickle

st.set_page_config(page_title="Random Forest Analyzer", layout="wide")
st.title("Random Forest Model")
st.write("อัพโหลดไฟล์ CSV เพื่อสร้างโมเดล Random Forest")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        return None

uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("ข้อมูลที่อัพโหลด")
        st.write(f"จำนวนแถว: {df.shape[0]}, จำนวนคอลัมน์: {df.shape[1]}")
        st.dataframe(df.head())
    
        st.subheader("ข้อมูลสถิติเบื้องต้น")
        st.write(df.describe())
        
        st.subheader("ข้อมูลที่หายไป")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'จำนวน': missing_data,
            'เปอร์เซ็นต์': missing_percent
        })
        st.dataframe(missing_df)
        
        st.header("สร้างโมเดล Random Forest")
        
        target_column = st.selectbox("เลือกคอลัมน์เป้าหมาย (Target)", df.columns)
        
        if target_column:
            feature_columns = st.multiselect("เลือกคอลัมน์ Feature", 
                                           [col for col in df.columns if col != target_column],
                                           default=[col for col in df.columns if col != target_column])
            
            is_numeric_target = pd.api.types.is_numeric_dtype(df[target_column])
            problem_type = "regression" if is_numeric_target else "classification"
            
            st.write(f"ประเภทของปัญหา: **{problem_type}**")
            
            st.subheader("ตั้งค่าพารามิเตอร์")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("จำนวน Trees", 10, 500, 100, 10)
                max_depth = st.slider("ความลึกสูงสุดของ Trees", 1, 100, 10)
            with col2:
                min_samples_split = st.slider("จำนวนตัวอย่างขั้นต่ำที่ต้องการสำหรับการแบ่ง", 2, 20, 2)
                test_size = st.slider("สัดส่วนข้อมูลทดสอบ", 0.1, 0.5, 0.2, 0.05)
            
            if st.button("สร้างโมเดล"):
                try:
                    df_model = df.copy()
                    for col in feature_columns + [target_column]:
                        if df_model[col].isnull().sum() > 0:
                            if pd.api.types.is_numeric_dtype(df_model[col]):
                                df_model[col].fillna(df_model[col].mean(), inplace=True)
                            else:
                                df_model[col].fillna(df_model[col].mode()[0], inplace=True)
                    
                    X = df_model[feature_columns].copy()
                    y = df_model[target_column].copy()
                    
                    label_encoders = {}
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        label_encoders[col] = le
                    
                    if not is_numeric_target and pd.api.types.is_object_dtype(y):
                        y_le = LabelEncoder()
                        y = y_le.fit_transform(y)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    with st.spinner('กำลังสร้างโมเดล...'):
                        if problem_type == "classification":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42
                            )
                        else:
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42
                            )
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    st.subheader("ผลการประเมินโมเดล")
                    if problem_type == "classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"ความแม่นยำ (Accuracy): {accuracy:.4f}")
                        
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)
                        
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                    else:
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        st.write(f"R² Score: {r2:.4f}")
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
                        
                        st.subheader("Actual vs Predicted")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.xlabel('Actual')
                        plt.ylabel('Predicted')
                        st.pyplot(fig)
                    
                    st.subheader("ความสำคัญของ Feature")
                    feature_importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    plt.title('Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("ทดลองทำนายข้อมูลใหม่")
                    st.write("ป้อนค่าใหม่เพื่อทำนาย:")
                    
                    new_data = {}
                    cols = st.columns(3)
                    for i, feature in enumerate(feature_columns):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if pd.api.types.is_numeric_dtype(df[feature]):
                                new_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
                            else:
                                options = df[feature].unique().tolist()
                                new_data[feature] = st.selectbox(f"{feature}", options)
                    
                    if st.button("ทำนาย"):
                        new_df = pd.DataFrame([new_data])
                        
                        for col in new_df.select_dtypes(include=['object']).columns:
                            if col in label_encoders:
                                new_df[col] = label_encoders[col].transform(new_df[col])
                        
                        prediction = model.predict(new_df)
                        
                        st.subheader("ผลการทำนาย")
                        if problem_type == "classification" and not is_numeric_target:
                            predicted_class = y_le.inverse_transform([int(prediction[0])])[0]
                            st.success(f"ผลการทำนาย: {predicted_class}")
                        else:
                            st.success(f"ผลการทำนาย: {prediction[0]:.4f}")
                
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการสร้างโมเดล: {e}")
                    st.exception(e)

with st.expander("ข้อมูลเพิ่มเติมเกี่ยวกับ Random Forest"):
    st.write("""
    ## Random Forest คืออะไร?
    Random Forest เป็นอัลกอริทึมการเรียนรู้แบบ ensemble ที่ใช้เทคนิคการรวม decision trees หลายต้นเข้าด้วยกัน
    
    ### ข้อดีของ Random Forest:
    - ป้องกันปัญหา overfitting
    - ทำงานได้ดีกับข้อมูลที่มีหลายมิติ
    - จัดการกับค่าที่หายไปได้ดี
    - ประเมินความสำคัญของคุณลักษณะต่างๆ ได้
    - ใช้ได้ทั้งงาน classification และ regression
    - ไม่ต้องการการปรับแต่งพารามิเตอร์มากนัก
    - ทนทานต่อ outliers และ noise
    
    ### พารามิเตอร์ที่สำคัญ:
    - n_estimators: จำนวน trees ในป่า
    - max_depth: ความลึกสูงสุดของแต่ละ tree
    - min_samples_split: จำนวนตัวอย่างขั้นต่ำที่ต้องการสำหรับการแบ่ง node
    """)

st.title("Logistic Regression Model")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("1. Data Preview")
    st.write("Data shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("2. Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.write("Missing Values:")
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({"Missing Values": missing_data, "Percentage (%)": (missing_data / len(df)) * 100})
    st.dataframe(missing_df)

    st.subheader("3. Exploratory Data Analysis")
    target_column = st.selectbox("Select target variable:", df.columns.tolist())
    
    if df[target_column].dtype == 'object' or df[target_column].nunique() <= 10:
        st.write("Target variable distribution:")
        fig, ax = plt.subplots()
        df[target_column].value_counts().plot(kind='bar', ax=ax)
        plt.title(f"Distribution of {target_column}")
        st.pyplot(fig)
    else:
        st.warning("Target variable should be categorical for Logistic Regression.")
        df[target_column] = (df[target_column] > df[target_column].median()).astype(int)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_features = st.multiselect("Select independent variables:", numeric_columns, default=numeric_columns[:5])
    
    if selected_features:
        st.write("Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[selected_features + [target_column]].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("4. Data Preparation and Modeling")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    X = df[selected_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    do_scaling = st.checkbox("Standardize features", value=True)
    if do_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    st.subheader("5. Logistic Regression Model")
    C_value = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.1)
    max_iter = st.slider("Maximum iterations", 100, 1000, 200, 50)
    solver = st.selectbox("Solver", ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'])
    
    model = LogisticRegression(C=C_value, max_iter=max_iter, solver=solver, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("6. Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    st.pyplot(fig)

    st.subheader("7. Model Coefficients")
    coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_[0]}).sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df)
    
    st.subheader("8. Prediction with New Data")
    new_data = {feature: st.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())) for feature in selected_features}
    new_df = pd.DataFrame([new_data])
    if do_scaling:
        new_df = scaler.transform(new_df)
    pred = model.predict(new_df)
    pred_prob = model.predict_proba(new_df)[0, 1]
    st.write(f"Prediction: Class {pred[0]}")
    st.write(f"Probability: {pred_prob:.4f}")

    st.subheader("9. Download Results")
    result_csv = coef_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Model Coefficients", result_csv, "logistic_regression_coefficients.csv", "text/csv")
    model_bytes = pickle.dumps(model)
    st.download_button("Download Trained Model", model_bytes, "logistic_regression_model.pkl", "application/octet-stream")
    
with st.expander("ข้อมูลเพิ่มเติมเกี่ยวกับ Logistic Regression"):
    st.write("""
    ## Logistic Regression คืออะไร?
    Logistic Regression เป็นอัลกอริทึมที่ใช้สำหรับการจำแนกประเภทโดยใช้ฟังก์ชัน Sigmoid เพื่อแปลงค่าผลลัพธ์ให้อยู่ในช่วง 0-1
    
    ### ข้อดีของ Logistic Regression:
    - เข้าใจง่ายและตีความผลลัพธ์ได้ง่าย
    - ใช้ได้ดีสำหรับข้อมูลที่มีความสัมพันธ์เชิงเส้น
    - คำนวณได้รวดเร็วและมีประสิทธิภาพ
    - สามารถใช้เป็น baseline model ก่อนใช้อัลกอริทึมที่ซับซ้อนขึ้น
    
    ### พารามิเตอร์ที่สำคัญ:
    - C: ค่าการปรับความซับซ้อนของโมเดล (ค่าเล็กช่วยลด overfitting)
    - solver: วิธีการหาคำตอบ เช่น ‘liblinear’, ‘saga’ เป็นต้น
    - max_iter: จำนวนรอบที่ใช้ในการฝึกโมเดล
    """)

