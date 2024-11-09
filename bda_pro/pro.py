import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"tweet_gpt.csv")

# Drop rows with missing 'clean_tweet' or 'sentiment' (used for the target variable)
df.dropna(subset=['clean_tweet', 'sentiment'], inplace=True)

# Replace '-' in 'sentiment_label' with actual sentiment values from 'sentiment'
df['sentiment_label'] = df['sentiment']

# Ensure the 'sentiment_label' column has multiple classes
if df['sentiment_label'].nunique() < 2:
    raise ValueError("The dataset contains only one class. Model training requires at least two classes.")

# Define features and target
X = df['clean_tweet']
y = df['sentiment_label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model on the entire training set
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'sentiment_model.pkl')

# Streamlit app
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Tweet Sentiment Analysis")
st.markdown("""
This application uses Machine Learning to predict the **sentiment** of a tweet.
Simply enter a tweet below and click **Predict Sentiment** to see the result.
""")

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Text input for user to predict sentiment
st.write("### Enter a tweet for sentiment prediction:")
user_input = st.text_area("")

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        # Predict sentiment
        prediction = model.predict([user_input])[0]
        
        # Display sentiment result with color-coding
        st.write("### The predicted sentiment is:")
        if prediction.lower() == "positive":
            st.markdown(f"<h2 style='color: green;'>😊 {prediction}</h2>", unsafe_allow_html=True)
        elif prediction.lower() == "negative":
            st.markdown(f"<h2 style='color: red;'>😞 {prediction}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: blue;'>😐 {prediction}</h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text for prediction.")

# Display classification report on the original test data
st.write("### Model Performance Metrics")
if st.button("Show Model Performance"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"**Model Accuracy:** {accuracy:.2%}")
    
    # Display classification report as a DataFrame for cleaner presentation
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color="lightgreen"))

    # Optionally, show metrics with additional color
    st.write("### Classification Report Summary:")
    st.write("##### Precision, Recall, F1-Score")
    st.text("Showing metrics per sentiment class:")
    st.table(report_df[['precision', 'recall', 'f1-score']])
