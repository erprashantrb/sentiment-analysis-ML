import streamlit as st
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Load updated models (from Models folder)
predictor = joblib.load("Models/lr_sentiment_model.pkl")
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")

# NLP setup
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))

# Streamlit UI setup
st.set_page_config(
    page_title="Sentilytics ️",
    page_icon="SentilyticsBYCodeCrafters",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg,#FFC1CC,#FFE680,#C1F5FF);
            border-radius:20px;
            padding:2rem;
            box-shadow:0 4px 20px rgba(0,0,0,0.1);
        }
        .stButton>button{
            background:#FF5E79;
            color:white;
            font-size:1.1rem;
            padding:0.8em 1.5em;
            border-radius:12px;
            transition:background 0.3s;
        }
        .stButton>button:hover{
            background:#E04E69;
        }
        .footer{
            text-align:center;
            margin-top:3rem;
            font-size:0.9rem;
            color:#333;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main">
  <h1 style="text-align:center; color:#A62E4E;">🎭 Sentilytics: Emotion Explorer</h1>
  <p style="text-align:center; font-size:1.2rem;">
     Decode feelings behind words
  </p>
</div>
""", unsafe_allow_html=True)

# Preprocessing

def preprocess(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)

# Columns
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📄 Upload CSV with 'Sentence' column", type="csv", key="csv")
    text_input = st.text_area("📜 Or type a sentence...", key="text", height=120)

with col2:
    st.write("")
    st.write("")
    if st.button("✨ Predict the Emotion"):
        if text_input.strip():
            processed = preprocess(text_input)
            X = vectorizer.transform([processed]).toarray()
            pred = predictor.predict(X)[0]
            sentiment = "Positive" if pred == 1 else "Negative"
            emoji = "😊" if pred == 1 else "😔"
            st.markdown(f"<h2 style='color:#A62E4E;'>🎯 Prediction: <strong>{sentiment}</strong> {emoji}</h2>", unsafe_allow_html=True)

        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "Sentence" not in df.columns:
                st.error("❌ CSV must include a 'Sentence' column.")
            else:
                df["Processed"] = df["Sentence"].apply(preprocess)
                X = vectorizer.transform(df["Processed"]).toarray()
                df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictor.predict(X)]

                st.success("✅ Bulk prediction complete!")
                st.dataframe(df[["Sentence", "Sentiment"]], height=300)

                fig, ax = plt.subplots()
                df["Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#6EFFA3", "#FF9F9F"], ax=ax)
                ax.set_title("Sentiment Distribution")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")

        else:
            st.warning("⚠️ Enter a sentence or upload a CSV file to analyze!")

# Romantic footer
st.markdown("""
<div class="footer">
  Made with warm code 💻 and warmer heart ❤️<br>
  Crafted by Prajakta🌺 & Prashant⭐ With  ❤️<
</div>
""", unsafe_allow_html=True)
