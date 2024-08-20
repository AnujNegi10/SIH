import streamlit as st
from model import get_model
from data_preprocessing import preprocess_text

# Load model, vectorizer, and PCA
model, vectorizer, pca = get_model()





# Streamlit app
st.title("Disaster Text Classification")
input_text = st.text_area("Enter text to analyze")

if st.button("Predict"):
    preprocessed_text = preprocess_text(input_text)
    text_vectorized = vectorizer.transform([preprocessed_text]).toarray()
    text_pca = pca.transform(text_vectorized)
    prediction = model.predict(text_pca)
    
    if prediction == 1:
        st.write("Prediction: **Disaster-related**")
        # Extract location from the text
    else:
        st.write("Prediction: **Not disaster-related**")


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/70573/fireman-firefighter-rubble-9-11-70573.jpeg?auto=compress&cs=tinysrgb&w=600");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
