import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint():

    # Case 1: Sample input data
    input_data = {
        "input_data_list": ["walter Metall cnc Bohrer",
                            "individuelle verpackung",
                            "402544096",
                            "konfektionierung faltschachteln",
                            "Stahlbau (DIN EN 1090-3 Niedersachsen",
                            "anodieren titan"]
    }

    # Send a POST request to the /predict endpoint with the sample data
    response = client.post("/predict", json=input_data)

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200

    # Parse the JSON response
    response_data = json.loads(response.text)
    response_data = json.loads(response_data)

    # Extract predicted labels from response_data
    predicted_labels = [item["Predicted_Label"] for item in response_data]
    # Check if any predicted label is in the specified list
    assert any(label in ["ch", "cnc", "ct", "ft", "mr", "pkg"] for label in predicted_labels)


def test_predict_endpoint_empty_input():
    # Case 2: Empty input data
    empty_input_data = {
        "input_data_list": []
    }

    # Send a POST request with empty input data
    empty_response = client.post("/predict", json=empty_input_data)

    # Check that the response status code is 400 for the empty input data
    assert empty_response.status_code == 400
