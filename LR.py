pip install scikit-learn
venv\Scripts\activate
pip show scikit-learn
streamlit run LR.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Linear Regression Analysis")

    # Upload file
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Determine file type and read into DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')

            # Show the uploaded data
            st.subheader("Uploaded Data")
            st.write(df)

            # Select columns for X (independent variables) and y (dependent variable)
            st.subheader("Select X and y columns")
            feature_cols = st.multiselect("Select X columns", df.columns)
            target_col = st.selectbox("Select y column", df.columns)

            # Check if columns are selected
            if feature_cols and target_col:
                X = df[feature_cols].values
                y = df[target_col].values

                # Perform linear regression
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                # Display metrics
                st.subheader("Linear Regression Results")
                st.write(f"Coefficients: {model.coef_}")
                st.write(f"Intercept: {model.intercept_}")
                st.write(f"Mean squared error: {mean_squared_error(y, y_pred):.2f}")
                st.write(f"R2 score: {r2_score(y, y_pred):.2f}")

                # Plot actual vs. predicted values (for univariate case)
                if len(feature_cols) == 1:
                    st.subheader("Actual vs. Predicted")
                    df_plot = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
                    st.line_chart(df_plot)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
