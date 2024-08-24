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
data = pd.read_csv(r'.\trainDisaster.csv')
data['text'] = data['text'].apply(preprocess_text)

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
