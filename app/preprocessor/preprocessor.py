import numpy as np
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    def __init__(self, data, column):
        try:
            self.processed_data = self.preprocess(data.copy(), column)
        except Exception as e:
            # Handle unexpected errors during preprocessing
            logging.error(f"Error during preprocessing: {e}")
            self.processed_data = None

    def preprocess(self, data, column):
        try:
            # Preprocess text
            # Remove unwanted text from the text column like email addresses, URLs, etc.
            data[column] = data[column].apply(self.remove_unwanted)
            # Remove special characters from the text column except German letters, digits, and whitespace
            data[column] = data[column].apply(self.remove_special_characters)
            # Preprocess German text by tokenizing, removing stopwords, and stemming
            data[column] = data[column].apply(self.preprocess_german_text)
            # Remove leading and trailing whitespaces
            data[column] = data[column].str.strip()
            # Remove rows with empty text and also text with length of text <= 1
            data.loc[data['text'].apply(lambda x: len(self.preprocess_german_text(x)) <= 1), 'text'] = np.nan
            for col in data.columns:
                data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)
            # Fill empty text with 'OTHERS' for any unseen text
            data.fillna({'text': 'OTHERS'}, inplace=True)
            data.reset_index(drop=True, inplace=True)

            return data

        except Exception as preprocessing_error:
            # Handle errors during preprocessing
            logging.error(f"Error during text preprocessing: {preprocessing_error}")
            return None

    @staticmethod
    def remove_unwanted(input_text):
        try:
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

        except Exception as remove_unwanted_error:
            # Handle errors during unwanted text removal
            logging.error(f"Error during unwanted text removal: {remove_unwanted_error}")
            return ''

    @staticmethod
    def remove_special_characters(input_text):
        try:
            # Define a regular expression pattern to match non-German letters, digits, and whitespace
            pattern = re.compile(r'[^a-zA-ZäöüßÄÖÜ0-9\s]')
            # Use the pattern to replace non-matching characters with an empty string
            cleaned_text = pattern.sub('', input_text)

            return cleaned_text

        except Exception as remove_special_characters_error:
            # Handle errors during special character removal
            logging.error(f"Error during special character removal: {remove_special_characters_error}")
            return ''

    @staticmethod
    def preprocess_german_text(input_text):
        try:
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

        except Exception as preprocessing_german_text_error:
            # Handle errors during German text preprocessing
            logging.error(f"Error during German text preprocessing: {preprocessing_german_text_error}")
            return ''
