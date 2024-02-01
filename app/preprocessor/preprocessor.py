import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocesor:
    def __init__(self, data, column):
        self.processed_data = self.preprocess(data.copy(), column)

    def preprocess(self, data, column):
        # Preprocess text
        data[column] = data[column].apply(self.remove_unwanted)
        data[column] = data[column].apply(self.remove_special_characters)
        data[column] = data[column].apply(self.preprocess_german_text)
        data[column] = data[column].str.strip()
        data.loc[data['text'].apply(lambda x: len(self.preprocess_german_text(x)) <= 1), 'text'] = np.nan
        for col in data.columns:
            data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)
        data.fillna({'text': 'OTHERS'}, inplace=True)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    def remove_unwanted(self, input_text):
        # Remove email addresses
        input_text = re.sub(r'\S+@\S+', '', input_text)
        # Remove HTML tags
        input_text = re.sub(r'<.*?>', '', input_text)
        # Remove URLs
        input_text = re.sub(r'http\S+|www\S+|https\S+', '', input_text)
        # Remove digits
        input_text = re.sub(r'\d+', '', input_text)

        input_text = input_text.strip()

        return input_text

    def remove_special_characters(self, input_text):
        # Define a regular expression pattern to match non-German letters, digits, and whitespace
        pattern = re.compile(r'[^a-zA-ZäöüßÄÖÜ0-9\s]')
        # Use the pattern to replace non-matching characters with an empty string
        cleaned_text = pattern.sub('', input_text)

        return cleaned_text


    def preprocess_german_text(self, input_text):
        lower_text = input_text.lower()
        # Tokenization
        tokens = word_tokenize(lower_text, language='german')
        # Removal of stopwords
        stop_words = set(stopwords.words('german'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
        # Stemming
        stemmer = SnowballStemmer('german')
        # Filter out single letters if they are standalone tokens
        tokens = [stemmer.stem(token) if len(token) > 1 else token for token in tokens]

        return ' '.join(tokens)