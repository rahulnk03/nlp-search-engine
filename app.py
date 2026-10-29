from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# 1. OUR LECTURE NOTES (The Corpus)
documents = [
    {"id": 1, "title": "Intro to Machine Learning", "content": "Machine learning involves training models using data. Supervised learning uses labeled data and backpropagation."},
    {"id": 2, "title": "Deep Learning Basics", "content": "Deep learning uses artificial neural networks with many hidden layers. It is highly effective for image recognition and computer vision."},
    {"id": 3, "title": "Natural Language Processing", "content": "NLP allows computers to understand human text. TF-IDF, tokenization, and word embeddings are foundational techniques in NLP."},
    {"id": 4, "title": "History of Rome", "content": "The Roman Empire was one of the largest in history. Julius Caesar was a famous Roman leader who crossed the Rubicon."}
]

corpus = [doc['content'] for doc in documents]

# 2. THE NLP MATH (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# 3. THE SEARCH ROUTINE
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in ranked_indices:
        if similarities[idx] > 0:
            results.append({
                "title": documents[idx]["title"],
                "snippet": documents[idx]["content"][:80] + "...",
                "score": round(similarities[idx], 4)
            })
            
    return jsonify(results)

if __name__ == '__main__':
    print("Search Engine is running on http://127.0.0.1:5000")
    app.run(debug=True)