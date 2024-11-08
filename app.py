import os
import streamlit as st
import pickle
import time
import pandas as pd
from model import get_models
from data_preprocessing import preprocess_text
import spacy
import re
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

with open("ensemble_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# with open("vectorizer.pkl", "rb") as vec_file:
#     vectorizer = pickle.load(vec_file)

# with open("pca.pkl", "rb") as pca_file:
#     pca = pickle.load(pca_file)

# Load model, vectorizer, PCA, and accuracy
vectorizer, pca, acc = get_models()

# Initialize spaCy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download the model if it hasn't been installed
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_sm")

# Define disaster keywords
DISASTER_KEYWORDS = {
    "FLOOD": [
        "flood", "flash floods", "heavy rains", "water level rise", "inundation",
        "deluge", "high water", "overflow", "flooding"
    ],
    "EARTHQUAKE": [
        "earthquake", "tremor", "quake", "seismic activity", "aftershock",
        "magnitude", "epicenter", "seismic event"
    ],
    "FOREST FIRE": [
        "forest fire", "wildfire", "blaze", "brush fire", "bushfire", "firestorm",
        "flame", "conflagration", "inferno"
    ],
    "LANDSLIDE": [
        "landslide", "mudslide", "rockslide", "earth slump", "debris flow", 
        "soil erosion", "land slip", "rockfall"
    ],
    "VOLCANIC ERUPTION": [
        "volcanic eruption", "lava flow", "volcano explosion", "ash cloud", 
        "pyroclastic flow", "volcanic activity", "lava eruption", "volcanic ash"
    ],
    "TSUNAMI": [
        "tsunami", "sea wave", "ocean wave", "tidal wave", "wave surge", 
        "coastal flood", "sea surge", "undersea quake"
    ],
    "HURRICANE": [
        "hurricane", "typhoon", "cyclone", "tropical storm", "storm surge",
        "category 5", "hurricane force", "high winds", "tropical cyclone"
    ],
    "TORNADO": [
        "tornado", "twister", "cyclone", "whirlwind", "funnel cloud",
        "tornado warning", "tornado watch", "vortex"
    ],
    "BLIZZARD": [
        "blizzard", "snowstorm", "whiteout", "heavy snow", "severe snowstorm",
        "snow squall", "snowdrift", "winter storm"
    ],
    "DROUGHT": [
        "drought", "dry spell", "water shortage", "arid conditions", "water scarcity",
        "low rainfall", "desiccation", "extended dry period"
    ]
}

# Helper functions
def extract_loc(text):
    doc = nlp(text)
    location = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return location

def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ["EVENT", "LOC", "GPE", "ORG"]]
    return keywords

def tell_time(text):
    doc = nlp(text)
    time = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    if not time:
        regex = r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]\s?(AM|PM|am|pm)?\b'
        time = re.findall(regex, text)
        time = [''.join(t) for t in time]
    return time 

def extract_date(text):
    doc = nlp(text)
    date = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    return date

def extract_detailplace(text):
    doc = nlp(text)
    place = [ent.text for ent in doc.ents if ent.label_ == "FAC"]
    return place

def detect_disaster_type(text):
    text_lower = text.lower()
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return disaster
    return None
st.markdown(
    """
    <style>
    .header {
        # position: absolute;
        top: 0px;
        left: 0px;
        color: red;
        font-size: 24px;
        font-weight: bold;
        z-index: 1;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="header">NDRF Dashboard</div>', unsafe_allow_html=True)


# Load the CSV file




# Reference the CSV file in the static folder
file_path = os.path.join("static", "trainDisaster.csv")
df = pd.read_csv(file_path)
# df = pd.read_csv(r"./trainDisaster.csv")
# uploaded_file = st.file_uploader("./trainDisaster.csv", type="csv")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

# Streamlit app
st.subheader("Automated Disaster Text Classification")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []
if 'current_index' not in st.session_state:
    st.session_state['current_index'] = 0

def process_and_predict(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text]).toarray()
    text_pca = pca.transform(text_vectorized)
    
    prediction = model.predict(text_pca)[0]
    
    if prediction == 1:
        location = extract_loc(text) or ["No Location Found"]
        time = tell_time(text) or ["No specific time mentioned"]
        date = extract_date(text) or ["No date mentioned"]
        detailplace = extract_detailplace(text) or ["No place Found"]
        keywords = extract_keywords(text) or ["No IMP keywords"]
        disaster_type = detect_disaster_type(text) or "Unknown Disaster"

        result = {
            'text': text,
            'prediction': "DISASTER-RELATED-CONTENT",
            'disaster_type': disaster_type,
            'location': ",".join(location) if location else "No Location Found",
            'time': " ".join(time) if time else "No specific time mentioned",
            'date': " ".join(date) if date else "No date mentioned",
            'detailplace': ", ".join(detailplace) if detailplace else "No place Found",
            'keywords': ", ".join(keywords) if keywords else "No IMP keywords",
            'accuracy': np.round(acc * 100, 1)
        }
    else:
        result = {
            'text': text,
            'prediction': "NOT DISASTER RELATED"
        }
    
    return result

# Loop through the CSV file every 10 seconds
for i in range(st.session_state['current_index'], len(df)):
    text = df.iloc[i]['text']  # Replace 'text_column_name' with your actual column name
    prediction_result = process_and_predict(text)
    
    # Append result to session state
    st.session_state['predictions'].append(prediction_result)
    
    # Update the current index
    st.session_state['current_index'] += 1
    
    # Display the result
    with st.container():
        if prediction_result['prediction'] == "DISASTER-RELATED-CONTENT":
            st.markdown(
                f"""
                <div class="prediction-box">
                    <strong>Prediction:</strong> {prediction_result['prediction']}<br>
                    {'News:  ' +  prediction_result['text'] + '<br>''<br>' if 'text' in prediction_result else ''}
                    {'Disaster Type:  '+  prediction_result['disaster_type'] + '<br>''<br>' if 'disaster_type' in prediction_result else ''}
                    {'Disaster Location:  '+  prediction_result['location'] + '<br>' if 'location' in prediction_result else ''}
                    {'Major Impact Seen At:  ' +  prediction_result['detailplace'] + '<br>' if 'detailplace' in prediction_result else ''}
                    {'Disaster Time:  ' +  prediction_result['time'] + '<br>' if 'time' in prediction_result else ''}
                    {'Held On Date:  ' +  prediction_result['date'] + '<br>' if 'date' in prediction_result else ''}
                    {'IMP Keywords:  ' +  prediction_result['keywords'] + '<br>' if 'keywords' in prediction_result else ''}
                    {'Accuracy:  ' +  str(prediction_result['accuracy']) + '%<br>' if 'accuracy' in prediction_result else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write(f"Content not disaster-related.")

        st.markdown("""
    <style>
    .prediction-box {
        background-color: black;
        color: white;
        border: 2px solid yellow;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 10px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.7);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 255, 0, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)
        
        

    
    # Wait for 10 seconds
    time.sleep(7)
    
    # Clear text area for next input
    st.empty()
