import requests

# Define the URL of your Flask API
url = 'http://localhost:5000/predict'  # Update the URL if needed

# Prepare the data to be sent as a JSON payload
data = {
    'sepal length': 3.57,  
    'sepal width': 7.57,     
    'petal length': 1.7708,
    'petal width' : 1.961
}

# Send a POST request to the Flask API
response = requests.post(url, json=data)

# Print the response from the API
print(response.json())