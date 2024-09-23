import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    store = pd.read_csv(r"superstore_final_dataset (1).csv", encoding='latin-1')

    store = store.dropna(subset=['Sales'])
    store = store.drop(['Row_ID', 'Order_ID', 'Customer_ID', 'Customer_Name', 'Postal_Code', 'Ship_Mode', 'Country', 'Ship_Date', 'Region'], axis=1)

    store['Product_of_Interest'] = store['Sales'] > 150

    X = store.drop(columns=['Sales', 'Product_of_Interest'])
    y = store['Product_of_Interest']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    logistic_model = LogisticRegression(random_state=42, max_iter=1000, solver='saga')

    logistic_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', logistic_model)
    ])

    logistic_model_pipeline.fit(X_train, y_train)

    return logistic_model_pipeline, X_test, y_test

# Load and preprocess data
model_pipeline, X_test, y_test = load_and_preprocess_data()

# Streamlit app
st.title('Superstore Sales Analysis and Prediction')

st.sidebar.header('User Input Features')

# Create user input fields for all features
user_inputs = {}
for column in X_test.columns:
    if X_test[column].dtype == 'object':
        unique_values = X_test[column].unique()
        user_inputs[column] = st.sidebar.selectbox(f"Select {column}", options=unique_values.tolist())
    else:
        user_inputs[column] = st.sidebar.number_input(f"Enter {column}", value=0)

# Create a DataFrame for the user input
user_df = pd.DataFrame([user_inputs])

# Predict
if st.button('Predict'):
    # Preprocess the user input
    user_proba = model_pipeline.predict_proba(user_df)[:, 1]
    
    # Determine the prediction category based on probability
    if user_proba[0] < 0.3:
        prediction_label = 'Low'
    elif user_proba[0] < 0.7:
        prediction_label = 'Medium'
    else:
        prediction_label = 'High'
    
    st.write(f"The predicted probability for 'Product_of_Interest' is: {user_proba[0]:.2f}")
    st.write(f"Prediction: {prediction_label}")

    # Model evaluation on test set
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = model_pipeline.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    st.subheader('Model Evaluation Metrics')
    st.write(f"AUC ROC Score: {auc_roc:.2f}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(class_report)

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    st.pyplot(plt)

# Additional descriptive insights based on sales data
st.subheader('Superstore Sales Insights')
st.write("""
    **Sales Revenue**: Tracking total sales revenue helps understand the financial performance and identify peak seasons or low sales periods.

    **Sales by Product Category**: Helps identify which categories drive revenue and which need improvement.

    **Sales by Region**: Identifies the best-performing stores or areas for expansion.

    **Sales by Customer Segments**: Inform targeted marketing strategies based on different customer segments.

    **Sales Trends**: Monthly or seasonal variations in sales to help with inventory management and marketing planning.

    **Product Performance**: Analyzing individual products to identify popular items or potential stockouts.

    **Customer Behavior**: Insights into customer preferences, purchase frequency, and loyalty.

    **Promotions and Discounts**: Evaluating the impact of promotions on sales to optimize marketing strategies.

    **Sales Forecasting**: Forecast future sales based on historical data for better inventory planning and decision-making.
""")
