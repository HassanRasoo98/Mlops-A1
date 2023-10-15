from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Titanic model
model = joblib.load('model/random_forest_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        slen = data['sepal length']
        swidth = data['sepal width']
        plen = data['petal length']
        pwidth = data['petal width']
        
        prediction = model.predict([[slen, swidth, plen, pwidth]])[0]
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)