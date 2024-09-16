from flask import Flask, request, render_template, redirect, url_for
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the stopwords corpus
nltk.download('stopwords')

app = Flask(__name__)

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

# Load the vectorizer and model
vector_form = pickle.load(open('vector1.pkl', 'rb'))
load_model = pickle.load(open('model1.pkl', 'rb'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    con = ' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    # Convert the sparse matrix to dense
    vector_form_dense = vector_form1.toarray()
    prediction = load_model.predict(vector_form_dense)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['news']
        prediction_class = fake_news(sentence)
        prediction = 'Reliable' if prediction_class == [0] else 'Unreliable'
        return redirect(url_for('result', prediction=prediction))

@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'unknown')
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
