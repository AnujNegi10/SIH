import streamlit as st
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
# Load model, vectorizer, PCA, and accuracy
model, vectorizer, pca, acc = get_models()

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_sm")

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

df = pd.read_csv("https://textinimage.s3.ap-south-1.amazonaws.com/trainDisaster.csv?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDEaCmFwLXNvdXRoLTEiSDBGAiEAsQAAAOvki8NMDF9MwY4ztOg75%2BthWJK2Ol8hwAF0UyICIQDc8LcFXl%2FpAOd3FrmCUOT5GwLA6sVHOnB1TqOO9dlaKirQAwiq%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDMzOTcxMjc4MjkwNyIMQccA08kobw9cpuYfKqQDFNUjWtmfnUqzqk3XS652f0Dp2bmuVXX1NO7MNF6jmaXETzGGYULgnSvYna%2Bv5bG0YLa%2BeI6x%2FIwJeQ4PMwBsjm%2FW3dFYMA%2BrYgzW55qgfe6VwEZ7%2B7y8CdA5hvUWvq3FGCtmgMPeh7cnaF4RhlRPzdx%2BksCC5jKBOeoRCICUtalU0B3oohUV8MvL2gEPNbyvyrrBTO5%2FS9Rx%2Fl7w9%2FNPUkWc%2BC3KzuyDWUzSGTRnRmHXsBipweDPZAfOwGBzBsvvEti%2B9NJWVWzHK%2FHYobZrsmbq5z64%2FGfhRK7bICu0J5eUwlGO9Q8cLke03oTzaRH1bTAQ%2BcnL26hfUmAmwyOtX7HbP5yqu8Iwbt8vgKeBTtBubU4KnQaPYR19MCGMIqQqA1%2FxQ6vTKJdHcLwinFfqAVFlLkR7AH72G1%2BG3Mn8FNLuo3UV%2FcYj3NeOqOdgSatanj8%2FT525DWbdEHVBmoWT8q2u9sTpm5zBtLXMCYWjUlA%2FZ3wsB203BDdnPMJrJYZU5K6NCM%2B6PJMUDobeMKQNm2Szv2aLDrFcu%2F34Xr066Ss3xskkMJyGlLkGOuMCOfwpPbqTvdktRt55DutxZxI91kPx%2FKVBpZBQtzj%2B48F2EBY0BB20mz4Tp2v1Obw9erpvwVoCxR5uLAkPX%2FSgYx0Y9DYc3CPBzkspSnvkRQgs6mOXlSJU%2BIh4U%2BkrGBFV%2BrMhxq8R7KdI3Rhx75Efv2fytKlyCEv2jBhCWQB8AVc5kHHL3CvGn%2BenqEmLB0%2Fg5ovyy2juGFOoFwyhYi7oUBT1%2FKPqoCgINtiGnb4YQqoz1FnbhdJhcfDoiuGidxU8Id2AOVveZWE7AWrnoJl6w24JJ11m12foYvA8J2CjPi6ziVjfwN4GGwC1LgeI0QiFd3aPfQWiUVvgrRG09vU5hGA%2FJcXcONsU2xUAn%2BkHWHialnv%2BYBFpxGRa0S3ZPZ9zWkU6nvICIxzZxEqFb1vx0xuFBwHSudtfil%2BQGGp1GxW14ibSVfnM9b2R5L8MUy7aKBvPxxcf1clNr2CThGLjK9eZHg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAU6GDVWI5VYGSID4O%2F20241101%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20241101T163718Z&X-Amz-Expires=18000&X-Amz-SignedHeaders=host&X-Amz-Signature=8e6b1bfab1ac298971be36b2f2a14355d565e5436448fbd09b3bee3a9658a0b5")
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
