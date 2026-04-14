# 🎬 Movie Recommendation System (Machine Learning)

A simple end-to-end Movie Recommendation System built with Machine Learning.

This project demonstrates how to go from raw data to a working API that recommends movies based on user profile similarities.

---

## 🚀 Overview

The system predicts the probability of a user liking a movie based on:

* User features (age, state)
* Movie features (genre, release year, duration)

It then returns the **Top N recommended movies** ranked by score.

---

## 🧠 How It Works

1. Load raw data (users, movies, interactions)
2. Apply feature engineering (encoding + normalization)
3. Train a neural network model
4. Generate predictions for all movies
5. Rank and return the best recommendations

---

## 🏗️ Architecture

```
Frontend → FastAPI → ML Model → Recommendations
```

Project structure:

```
src/
│
├── data/        # Data loading and preprocessing
├── model/       # Training and prediction logic
├── api/         # FastAPI application
└── main.py
```

---

## ⚙️ Tech Stack

* Python
* TensorFlow
* FastAPI
* NumPy
* Pandas

---

## 📊 Example Output

```json
{
  "usuario": "John",
  "recomendacoes": [
    { "titulo": "Movie A", "score": 0.82, "score_percent": "82%" },
    { "titulo": "Movie B", "score": 0.79, "score_percent": "79%" },
    { "titulo": "Movie C", "score": 0.75, "score_percent": "75%" }
  ]
}
```

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendations.git
cd movie-recommendations
```

---

### 2. Create virtual environment (Python 3.12 recommended)

```bash
py -3.12 -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the API

```bash
uvicorn src.api.app:app --reload
```

---

### 5. Access API docs

```
http://127.0.0.1:8000/docs
```

---

## 📥 API Usage

### POST `/recommend`

Request:

```json
{
  "nome": "John",
  "idade": 30,
  "estado": "SP"
}
```

Response:

```json
{
  "usuario": "John",
  "recomendacoes": [...]
}
```

---

## 🧪 Model Details

* Input: concatenated user + movie feature vectors
* Output: probability (0 to 1)
* Loss: Binary Crossentropy
* Task: Binary classification (like / not like)

---

## 💡 Key Learnings

* ML is not just about models — it's about pipelines
* Feature engineering is critical
* Clean architecture improves scalability
* Serving models via API is essential in real-world systems

---

## 🔮 Future Improvements

* Larger and more realistic dataset
* Embeddings (user/movie representation)
* Vector database integration (FAISS)
* Batch prediction optimization
* Model persistence (save/load instead of retraining)

---

## 🤝 Contributing

Feel free to open issues or submit pull requests.

---

## 📄 License

MIT

---

## 👨‍💻 Author

Built by Túlio Vitor

---

⭐ If you found this useful, consider giving it a star!


