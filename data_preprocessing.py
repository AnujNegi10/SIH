import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lemma = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text
