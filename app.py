# Import the necessary libraries
import warnings
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

data = pd.read_csv(r"C:\Users\Administrator\Desktop\project\datamlproject.csv")

# Ignore warnings
warnings.filterwarnings("ignore")

# Remove duplicates from the DataFrame
data = data.drop_duplicates()

# Combine the text columns
data['combined'] = data['title'] + ' ' + data['text'] + ' ' + data['subject'] + ' ' + data['date']

# Split the data into features and target
X = data['combined']
y = data['class']

# Initialize the vectorizer and scaler
vectorizer = TfidfVectorizer(tokenizer=lambda doc: [WordNetLemmatizer().lemmatize(t) for t in word_tokenize(doc) if t.lower() not in stopwords.words('english')])
standard_scaler = StandardScaler(with_mean=False)

# Transform the features using the vectorizer and scaler
X_tfidf = vectorizer.fit_transform(X)
X_scaled = standard_scaler.fit_transform(X_tfidf)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.40, random_state=42)

lr_model = LogisticRegression()

# Train and evaluate Logistic Regression model with cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
lr_cv_scores_mean = lr_cv_scores.mean()

# Train the Logistic Regression model on the entire training set
lr_model.fit(X_train, y_train)

# Create a function to predict news
def predict_news(user_input_news):
    vectorized_news = vectorizer.transform([user_input_news])
    scaled_news = standard_scaler.transform(vectorized_news)
    prediction = lr_model.predict(scaled_news)
    prediction_text = "Fake News" if prediction[0] == 1 else "Not Fake News"
    return prediction_text

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input_news = request.form['news_input']
        prediction_text = predict_news(user_input_news)
        return render_template('results.html', prediction_text=prediction_text)
    return render_template('input_news.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
