import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import preprocess_text

# Load and preprocess data
data = pd.read_csv(r'C:\Users\negia\OneDrive\Documents\data\trainDisaster.csv')
data['text'] = data['text'].apply(preprocess_text)

# Vectorize the text
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['text']).toarray()

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, data['target'], test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the vectorizer, PCA, and model as variables
def get_model():
    return model, vectorizer, pca
