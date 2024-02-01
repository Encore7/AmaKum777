import pandas as pd
import numpy as np
from app.preprocessor.preprocessor import Preprocesor
from app.feature.feature import FeatureMapper
import pickle
import tensorflow as tf

class Predictor:
    # Load the pickled model and vectorized
    classifier = tf.keras.models.load_model("./model_files/LSTM_model.h5")
    tokenizer = pickle.load(open("./model_files/tokenizer_mapper.pkl", "rb"))

    def __init__(self, input_data_df):
        self.predicted = self.predict_data(input_data_df)

    def predict_data(self, input_data_df):
        new_input_df_processed = Preprocesor(input_data_df, 'text')
        feature_df = FeatureMapper(new_input_df_processed.processed_data, 'text', self.tokenizer)
        predicted_prob = self.classifier.predict(feature_df.feature_data)
        prediction = np.argmax(predicted_prob, axis=1)
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