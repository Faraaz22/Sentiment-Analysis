from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle

app = Flask(__name__, template_folder='index.html')

# Enable CORS for the entire app
CORS(app)

# Load the trained sentiment model
with open('sentiment_model.sav', 'rb') as rf:
    model = pickle.load(rf)

# Load the vectorizer 
with open('vectorizer.sav', 'rb') as vf:
    vectorizer = pickle.load(vf)

@app.route('/', methods=['GET'])
def index():
    print("Loading.........")
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the tweet from the request data
    tweet = data['tweet']

    # Transform the tweet using the vectorizer
    tweet_vectorized = vectorizer.transform([tweet])

    # Use the model to predict the sentiment of the tweet
    sentiment = model.predict(tweet_vectorized)[0]  # 1 -> positive, 0 -> negative

    sentiment_result = 'Positive' if sentiment == 1 else 'Negative'

    # Return the result as JSON
    return jsonify({'sentiment': sentiment_result})

if __name__ == '__main__':
    app.run(debug=True)
