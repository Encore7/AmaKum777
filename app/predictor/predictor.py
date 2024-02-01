from app.preprocessor.preprocessor import Preprocessor
from app.feature.feature import FeatureMapper

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import logging
import sys
import os
import warnings

warnings.filterwarnings('ignore')


class Predictor:
    # Load the feature mapper and the model
    root_directory = os.getcwd()
    print(root_directory)
    try:
        classifier = tf.keras.models.load_model(f"{root_directory}/model_files/LSTM_model.h5")
    except Exception as model_loading_error:
        # Handle errors related to loading the model
        logging.error(f"Error loading the model: {model_loading_error}")
        sys.exit(1)
    try:
        tokenizer = pickle.load(open(f"{root_directory}/model_files/tokenizer_mapper.pkl", "rb"))
    except Exception as tokenizer_loading_error:
        # Handle errors related to loading the tokenizer
        logging.error(f"Error loading the tokenizer: {tokenizer_loading_error}")
        sys.exit(1)

    def __init__(self, input_data_df):
        self.predicted = self.predict_data(input_data_df)

    def predict_data(self, input_data_df):
        try:
            # Preprocessing the input data
            new_input_df_processed = Preprocessor(input_data_df, 'text')
            # Mapping the preprocessed data to features
            feature_df = FeatureMapper(new_input_df_processed.processed_data, 'text', self.tokenizer)
            # predicting with probability
            predicted_prob = self.classifier.predict(feature_df.feature_data)
            # predicting with labels
            prediction = np.argmax(predicted_prob, axis=1)
            # Create a dataframe with the input text and the predicted label
            results_df = pd.DataFrame({'Text': input_data_df["text"], 'Predicted_Label': prediction})

            # Map the predicted numeric labels to the desired labels
            results_df['Predicted_Label'] = results_df['Predicted_Label'].map({
                0: "ch",
                1: "cnc",
                2: "ct",
                3: "ft",
                4: "mr",
                5: "pkg"
            })

            return results_df
        except Exception as prediction_error:
            # Handle errors related to making predictions
            logging.error(f"Error making predictions: {prediction_error}")
            return pd.DataFrame({'Text': input_data_df["text"], 'Predicted_Label': None})
