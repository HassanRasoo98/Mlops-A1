import joblib
import pytest

# Load the pre-trained model
model = joblib.load('model/titanic_model.pkl')  # Replace with the correct path to your model

@pytest.fixture
def sample_input_data():
    # Define sample input data for testing
    return {
        'sepal_length': 2.86,
        'sepal_width': 3.55,
        'petal_length': 2.14,
        'petal_width': 3.97
    }

def test_model_prediction(sample_input_data):
    # Make predictions using the model
    prediction = model.predict([[sample_input_data['sepal_length'], sample_input_data['sepal_width'], sample_input_data['petal_length'], sample_input_data['petal_width']]])

    # Perform assertions on the prediction
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]  # Assuming your model predicts binary outcomes