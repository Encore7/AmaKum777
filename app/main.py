import pandas as pd
from fastapi import FastAPI
import uvicorn
from app.input.input_text import InputTextList
from app.predictor.predictor import Predictor

# Create the app object
app = FastAPI()

# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted label
@app.post("/predict")
async def predict_text_label(data: InputTextList):
    new_input_df = pd.DataFrame({"text": data.input_data_list})
    predicted_df = Predictor(new_input_df)

    return predicted_df.predicted.to_json(orient="records")

# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload
