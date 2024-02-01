from app.input.input_text import InputTextList
from app.predictor.predictor import Predictor

from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Create the app object
app = FastAPI()


@app.post("/predict")
async def predict(data: InputTextList):
    try:
        # Check if the input data is empty
        if not data.input_data_list:
            raise HTTPException(status_code=400, detail="Input data is empty")

        # creating a dataframe from the input data
        new_input_df = pd.DataFrame({"text": data.input_data_list})
        # Make predictions
        predicted_df = Predictor(new_input_df)

        return predicted_df.predicted.to_json(orient="records")

    except Exception as exception:
        # Handle unexpected errors
        return {"error": str(exception)}


# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
