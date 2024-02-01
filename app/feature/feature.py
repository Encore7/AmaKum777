from tensorflow.keras.preprocessing.sequence import pad_sequences

class FeatureMapper:
    def __init__(self, data, column, tokenizer):
        self.feature_data = self.make_feature(data, column, tokenizer)

    def make_feature(self, data, column, tokenizer):
        tokenize_df = tokenizer.texts_to_sequences(data[column].values)
        padded_df = pad_sequences(tokenize_df, maxlen=75)

        return padded_df
