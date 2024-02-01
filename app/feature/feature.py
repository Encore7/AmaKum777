from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import logging


class FeatureMapper:
    def __init__(self, data, column, tokenizer):
        try:
            self.feature_data = self.make_feature(data, column, tokenizer)
        except Exception as e:
            # Handle unexpected errors during feature mapping
            logging.error(f"Error during feature mapping: {e}")
            self.feature_data = None

    @staticmethod
    def make_feature(data, column, tokenizer):
        try:
            # Mapping the text from train data to features text
            tokenize_df = tokenizer.texts_to_sequences(data[column].values)
            # Padding the text to a maximum length of 75
            padded_df = pad_sequences(tokenize_df, maxlen=75)

            return padded_df

        except Exception as feature_mapping_error:
            # Handle errors during feature mapping
            logging.error(f"Error during feature mapping: {feature_mapping_error}")
            return np.zeros((len(data), 75))  # Return zeros if there's an error
