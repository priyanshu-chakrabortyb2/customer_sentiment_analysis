from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sqlite3

app = FastAPI(
    title="Customer Sentiment Analysis API",
    version="1.0"
)

model = joblib.load(
    "final_sentiment_model.pkl"
)

vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

class ReviewRequest(BaseModel):
    review_text: str

def create_table():

    conn = sqlite3.connect(
        "customer_sentiment.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            review_text TEXT NOT NULL,

            predicted_sentiment TEXT,

            model_name TEXT,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP

        )
        """
    )

    conn.commit()
    conn.close()


create_table()

def save_prediction(
    review_text,
    predicted_sentiment
):

    conn = sqlite3.connect(
        "customer_sentiment.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO reviews
        (
            review_text,
            predicted_sentiment,
            model_name
        )
        VALUES (?, ?, ?)
        """,
        (
            review_text,
            predicted_sentiment,
            "LinearSVC"
        )
    )

    conn.commit()
    conn.close()

@app.get("/")
def home():

    return {
        "message": "Customer Sentiment Analysis API"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(
    request: ReviewRequest
):

    try:

        review_vector = vectorizer.transform(
            [request.review_text]
        )

        prediction = model.predict(
            review_vector
        )[0]

        save_prediction(
            request.review_text,
            str(prediction)
        )

        return {

            "review_text":
            request.review_text,

            "predicted_sentiment":
            str(prediction)

        }

    except Exception as e:

        return {
            "error": str(e)
        }

@app.get("/reviews")
def get_reviews():

    conn = sqlite3.connect(
        "customer_sentiment.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
        id,
        review_text,
        predicted_sentiment,
        model_name,
        timestamp
        FROM reviews
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    reviews = []

    for row in rows:

        reviews.append({

            "id": row[0],

            "review_text": row[1],

            "predicted_sentiment": row[2],

            "model_name": row[3],

            "timestamp": row[4]

        })

    return reviews