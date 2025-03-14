import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# โหลดและแสดงผลข้อมูล
st.title("Regression Model with PyTorch in Streamlit")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview Data", df.head())

    # เลือก features และ target
    columns = list(df.columns)
    feature_columns = st.multiselect("Select Features", columns)
    target_column = st.selectbox("Select Target", columns)

    if feature_columns and target_column:
        X = df[feature_columns].values
        y = df[target_column].values.reshape(-1, 1)

        # แบ่งข้อมูล train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize ข้อมูล
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        # แปลงเป็น Tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # สร้างโมเดล Regression ด้วย PyTorch
        class RegressionModel(nn.Module):
            def __init__(self, input_dim):
                super(RegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return self.linear(x)

        model = RegressionModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # ฝึกโมเดล
        epochs = 100
        loss_list = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # แสดงกราฟ Loss
        st.write("### Training Loss")
        fig, ax = plt.subplots()
        ax.plot(range(epochs), loss_list, label="Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

        # ทำนายค่าบนชุดทดสอบ
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)

        y_pred = scaler_y.inverse_transform(y_pred_tensor.numpy())
        y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

        # แสดงผลการทำนาย
        st.write("### Prediction Results")
        results_df = pd.DataFrame({"Actual": y_test_actual.flatten(), "Predicted": y_pred.flatten()})
        st.write(results_df.head())

        # กราฟเปรียบเทียบ
        st.write("### Actual vs. Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test_actual, y_pred, alpha=0.5)
        ax.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], "r--")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)
