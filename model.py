import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import preprocess_text
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
data = pd.read_csv("https://textinimage.s3.ap-south-1.amazonaws.com/trainDisaster.csv?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDEaCmFwLXNvdXRoLTEiSDBGAiEAsQAAAOvki8NMDF9MwY4ztOg75%2BthWJK2Ol8hwAF0UyICIQDc8LcFXl%2FpAOd3FrmCUOT5GwLA6sVHOnB1TqOO9dlaKirQAwiq%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDMzOTcxMjc4MjkwNyIMQccA08kobw9cpuYfKqQDFNUjWtmfnUqzqk3XS652f0Dp2bmuVXX1NO7MNF6jmaXETzGGYULgnSvYna%2Bv5bG0YLa%2BeI6x%2FIwJeQ4PMwBsjm%2FW3dFYMA%2BrYgzW55qgfe6VwEZ7%2B7y8CdA5hvUWvq3FGCtmgMPeh7cnaF4RhlRPzdx%2BksCC5jKBOeoRCICUtalU0B3oohUV8MvL2gEPNbyvyrrBTO5%2FS9Rx%2Fl7w9%2FNPUkWc%2BC3KzuyDWUzSGTRnRmHXsBipweDPZAfOwGBzBsvvEti%2B9NJWVWzHK%2FHYobZrsmbq5z64%2FGfhRK7bICu0J5eUwlGO9Q8cLke03oTzaRH1bTAQ%2BcnL26hfUmAmwyOtX7HbP5yqu8Iwbt8vgKeBTtBubU4KnQaPYR19MCGMIqQqA1%2FxQ6vTKJdHcLwinFfqAVFlLkR7AH72G1%2BG3Mn8FNLuo3UV%2FcYj3NeOqOdgSatanj8%2FT525DWbdEHVBmoWT8q2u9sTpm5zBtLXMCYWjUlA%2FZ3wsB203BDdnPMJrJYZU5K6NCM%2B6PJMUDobeMKQNm2Szv2aLDrFcu%2F34Xr066Ss3xskkMJyGlLkGOuMCOfwpPbqTvdktRt55DutxZxI91kPx%2FKVBpZBQtzj%2B48F2EBY0BB20mz4Tp2v1Obw9erpvwVoCxR5uLAkPX%2FSgYx0Y9DYc3CPBzkspSnvkRQgs6mOXlSJU%2BIh4U%2BkrGBFV%2BrMhxq8R7KdI3Rhx75Efv2fytKlyCEv2jBhCWQB8AVc5kHHL3CvGn%2BenqEmLB0%2Fg5ovyy2juGFOoFwyhYi7oUBT1%2FKPqoCgINtiGnb4YQqoz1FnbhdJhcfDoiuGidxU8Id2AOVveZWE7AWrnoJl6w24JJ11m12foYvA8J2CjPi6ziVjfwN4GGwC1LgeI0QiFd3aPfQWiUVvgrRG09vU5hGA%2FJcXcONsU2xUAn%2BkHWHialnv%2BYBFpxGRa0S3ZPZ9zWkU6nvICIxzZxEqFb1vx0xuFBwHSudtfil%2BQGGp1GxW14ibSVfnM9b2R5L8MUy7aKBvPxxcf1clNr2CThGLjK9eZHg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAU6GDVWI5VYGSID4O%2F20241101%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20241101T163718Z&X-Amz-Expires=18000&X-Amz-SignedHeaders=host&X-Amz-Signature=8e6b1bfab1ac298971be36b2f2a14355d565e5436448fbd09b3bee3a9658a0b5")
# data['text'] = data['text'].apply(preprocess_text)
if 'text' in data.columns:
    data['text'] = data['text'].fillna('')  # Fill missing values with empty string
    data['text'] = data['text'].apply(preprocess_text)
else:
    raise ValueError("The 'text' column is missing from the dataset")
# Vectorize the text
vectorizer = CountVectorizer(max_features=3000,ngram_range=(1,3))
X = vectorizer.fit_transform(data['text']).toarray()

# Apply PCA
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, data['target'], test_size=0.2, random_state=42)

# Train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(subsample= 0.9, n_estimators= 400, min_child_weight= 3 , max_depth= 9, learning_rate= 0.01, gamma=0, colsample_bytree= 1.0)),
    ('rf', RandomForestClassifier()),
    ('ada', AdaBoostClassifier())
], voting='hard')
ensemble_model.fit(X_train,y_train)

y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)


# Save the vectorizer, PCA, and model as variables
def get_models():
    return ensemble_model , vectorizer, pca , acc
