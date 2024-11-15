# Twitter Sentiment Analysis

This project performs sentiment analysis on tweets using machine learning techniques. It allows users to input a tweet, and based on its text, it predicts whether the sentiment is positive or negative.

## Overview

The project consists of three main components:
1. **Frontend**: A simple web interface to input a tweet and display its sentiment.
2. **Backend**: A machine learning model trained on tweet data to predict sentiment (positive or negative).
3. **Dataset**: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140

## Table of Contents
- [Technologies Used](#technologies-used)
- [Setup](#setup)
- [Frontend (HTML)](#frontend-html)
- [Backend (Python)](#backend-python)
- [Model Training](#model-training)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [License](#license)

## Technologies Used

- **Frontend**: 
  - HTML, CSS (for styling), and JavaScript (for handling user input and interacting with the backend).
  
- **Backend**:
  - Python
  - `scikit-learn`: For machine learning.
  - `nltk`: For text preprocessing and stemming.
  - `tensorflow`: Optional for advanced models, but not used in this basic model.
  - `pickle`: For saving and loading the trained model and vectorizer.

## Setup

To set up this project locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a dataset of tweets in a CSV format, as used in the code (`tweets.csv`).

4. Start the backend server. If you are using Flask (as inferred from the code):

   ```bash
   flask run
   ```

   The backend will run locally on `http://127.0.0.1:5000/`.

5. Open `index.html` in your browser to use the sentiment analysis web app.

## Frontend (HTML)

The frontend is a simple HTML form that allows users to input a tweet. The form is part of the `index.html` file, which makes a request to the backend server when the user submits a tweet.

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Twitter Sentiment Analysis</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <h1>Twitter Sentiment Analysis</h1>
    <form id="twitter-form">
      <textarea id="tweet" placeholder="Enter your tweet..."></textarea>
      <button type="submit">Analyze</button>
    </form>
    <div id="sentiment-result"></div>

    <script>
      document.getElementById("twitter-form").addEventListener("submit", async function(event) {
        event.preventDefault();
        const tweetText = document.getElementById("tweet").value;

        if (!tweetText.trim()) {
          alert("Please enter a valid tweet.");
          return;
        }

        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ tweet: tweetText })
        });

        const data = await response.json();
        document.getElementById("sentiment-result").textContent = `Sentiment: ${data.sentiment}`;
      });
    </script>
  </body>
</html>
```

- Users can input text into the `textarea`.
- When the form is submitted, the text is sent to the backend for sentiment analysis.
- The result is displayed below the form (`Sentiment: Positive` or `Sentiment: Negative`).

## Backend (Python)

### Preprocessing

Before training the model, tweets undergo text preprocessing:
- **URL Removal**: Removes links (e.g., `http://...`).
- **Mentions and Hashtags Removal**: Strips `@user` and `#hashtag`.
- **Lowercasing**: Converts all text to lowercase for uniformity.
- **Stemming**: Uses the `nltk` Porter stemmer to reduce words to their root forms.

### Model Training

The model is trained using a dataset of labeled tweets with binary sentiment (0 = Negative, 1 = Positive). The following steps are involved in the training process:

1. **Loading Data**: The dataset is loaded using `pandas` and preprocessed.
2. **Text Vectorization**: `TfidfVectorizer` is used to convert the text into numerical features.
3. **Model Training**: A machine learning model, such as `MultinomialNB(accuracy - 76%(approx))` and  `Logistic Regression(accuracy - 78%(approx))`, is trained using the 
   processed text data.
5. **Model Evaluation**: The model's accuracy is evaluated using a test dataset.

### Example Python Code for Model Training:

```python
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load the data
df = pd.read_csv('tweets.csv', encoding='cp1252')

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)    # Remove mentions
    text = re.sub(r'#\S+', '', text)    # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()  # Convert text to lowercase
    return text

df['text'] = df['text'].apply(preprocess_text)

# Prepare the data for training
X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the model and vectorizer
with open('sentiment_model.sav', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.sav', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
```

### Model Evaluation

The accuracy score of the model is computed using the test set, and the trained model and vectorizer are saved to disk using `pickle` for later use.

## Usage

1. Launch the backend server.
2. Open the `index.html` file in a web browser.
3. Enter a tweet into the input field and click "Analyze".
4. The sentiment (positive or negative) will be displayed based on the tweet's text.

## Screenshots
<img width="952" alt="image" src="https://github.com/user-attachments/assets/2dcf2d56-40a1-4d15-9c7b-13b83136a820">
<img width="950" alt="image" src="https://github.com/user-attachments/assets/e829ede1-6152-41c9-989e-9a1fd8cad466">
<img width="950" alt="image" src="https://github.com/user-attachments/assets/972ad1d3-b170-47d6-8c74-4ab883491f64">
<img width="959" alt="image" src="https://github.com/user-attachments/assets/5a3ebf70-2544-49d9-bc85-87710feed48b">

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This documentation includes an overview of the project, technologies used, and how to set up and run the project locally. You can copy and paste this into a `README.md` file on GitHub and update the necessary URLs, and repository details as appropriate.



