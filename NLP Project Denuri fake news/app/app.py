from flask import Flask, render_template, request
import pickle, re, os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords"); nltk.download("wordnet")
app = Flask(__name__, template_folder="templates")

# Get the directory of this script
app_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(app_dir), "models")

# Load models
with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"),"rb") as f: tfidf = pickle.load(f)
with open(os.path.join(models_dir, "naive_bayes_model.pkl"),"rb") as f: nb_model = pickle.load(f)
cnn_model = tf.keras.models.load_model(os.path.join(models_dir, "cnn_model.h5"))
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method == "POST":
        news = request.form["news_text"]
        model_choice = request.form["model"]
        cleaned = clean(news)

        if model_choice == "nb":
            vec = tfidf.transform([cleaned])
            pred = nb_model.predict(vec)[0]
        else:
            ids = bert_tokenizer([cleaned], max_length=128,
                                  padding="max_length", truncation=True,
                                  return_tensors="np")["input_ids"]
            pred = int(cnn_model.predict(ids)[0][0] >= 0.5)

        result = "✅ REAL News" if pred == 1 else "🚨 FAKE News"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)