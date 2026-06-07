import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Config

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    layout="wide"
)

# Load Model
model = joblib.load(
    "final_sentiment_model.pkl"
)

vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)


# Load Database


def load_data():

    conn = sqlite3.connect(
        "customer_sentiment.db"
    )

    query = """
    SELECT *
    FROM reviews
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df



# Title


st.title(
    "📊 Customer Sentiment Analysis Dashboard"
)


# Prediction Section

st.subheader(
    "Predict Sentiment"
)

review = st.text_area(
    "Enter Customer Review"
)

if st.button(
    "Analyze Sentiment"
):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        review_vector = vectorizer.transform(
            [review]
        )

        prediction = model.predict(
            review_vector
        )[0]

        if prediction == "Positive":

            st.success(
                f"😊 Predicted Sentiment: {prediction}"
            )

        elif prediction == "Negative":

            st.error(
                f"😞 Predicted Sentiment: {prediction}"
            )

        else:

            st.warning(
                f"😐 Predicted Sentiment: {prediction}"
            )

# Load Reviews


df = load_data()


# KPI Cards

if not df.empty:

    total_reviews = len(df)

    positive_reviews = len(
        df[
            df["predicted_sentiment"]
            == "Positive"
        ]
    )

    negative_reviews = len(
        df[
            df["predicted_sentiment"]
            == "Negative"
        ]
    )

    neutral_reviews = len(
        df[
            df["predicted_sentiment"]
            == "Neutral"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Positive",
        positive_reviews
    )

    col3.metric(
        "Negative",
        negative_reviews
    )

    col4.metric(
        "Neutral",
        neutral_reviews
    )

# Pie Chart


st.subheader(
    "Sentiment Distribution"
)

sentiment_counts = (
    df["predicted_sentiment"]
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = [
    "Sentiment",
    "Count"
]

fig = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Sentiment Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Recent Predictions


st.subheader(
    "Recent Predictions"
)

display_df = df[
    [
        "id",
        "review_text",
        "predicted_sentiment",
        "model_name",
        "timestamp"
    ]
].copy()

display_df.columns = [
    "Review ID",
    "Review",
    "Predicted Sentiment",
    "Model",
    "Timestamp"
]

st.dataframe(
    display_df.sort_values(
        by="Review ID",
        ascending=False
    ),
    use_container_width=True
)

# Filter Reviews

st.subheader(
    "Filter Reviews"
)

selected_sentiment = st.selectbox(
    "Select Sentiment",
    ["All", "Positive", "Negative", "Neutral"]
)

filtered_df = df.copy()

if selected_sentiment != "All":

    filtered_df = df[
        df["predicted_sentiment"]
        == selected_sentiment
    ]

display_filtered_df = filtered_df[
    [
        "review_text",
        "predicted_sentiment",
        "timestamp"
    ]
].copy()

display_filtered_df.columns = [
    "Review",
    "Sentiment",
    "Timestamp"
]

st.dataframe(
    display_filtered_df.iloc[::-1],
    use_container_width=True
)

# Sentiment Bar Chart
st.subheader("Sentiment Count")

bar_fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    title="Sentiment Distribution"
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)


# Search Review 
# Search Reviews

st.subheader(
    "Search Reviews"
)

search_term = st.text_input(
    "Search Review Text"
)

if search_term:

    search_results = df[
        df["review_text"]
        .str.contains(
            search_term,
            case=False,
            na=False
        )
    ]

    if not search_results.empty:

        display_search_df = search_results[
            [
                "review_text",
                "predicted_sentiment",
                "timestamp"
            ]
        ].copy()

        display_search_df.columns = [
            "Review",
            "Sentiment",
            "Timestamp"
        ]

        st.dataframe(
            display_search_df,
            use_container_width=True
        )

    else:

        st.info(
            "No matching reviews found."
        )

# Review Length Analysis

df["review_length"] = (
    df["review_text"]
    .astype(str)
    .apply(len)
)

st.subheader(
    "Review Length Analysis"
)

length_fig = px.histogram(
    df,
    x="review_length",
    nbins=30,
    title="Review Length Distribution"
)

st.plotly_chart(
    length_fig,
    use_container_width=True
)

# Batch CSV Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    batch_df = pd.read_csv(
        uploaded_file
    )

    st.write(
        "Preview"
    )

    st.dataframe(
        batch_df.head()
    )

    if "Review Text" in batch_df.columns:

        vectors = vectorizer.transform(
            batch_df["Review Text"]
            .astype(str)
        )

        batch_df[
            "Predicted Sentiment"
        ] = model.predict(
            vectors
        )

        st.success(
            "Prediction Complete"
        )

        st.dataframe(
            batch_df.head()
        )

        csv = batch_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

# Word Cloud
st.subheader(
    "Word Cloud"
)

if not df.empty:

    text = " ".join(
        df["review_text"]
        .astype(str)
    )

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots()

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig)

else:

    st.info(
        "No reviews available for Word Cloud."
    )

# Sentiment Trend Analysis
df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

trend_df = (
    df.groupby(
        df["timestamp"].dt.date
    )
    .size()
    .reset_index(name="count")
)

st.subheader(
    "Prediction Trend"
)

trend_fig = px.line(
    trend_df,
    x="timestamp",
    y="count",
    markers=True
)

st.plotly_chart(
    trend_fig,
    use_container_width=True
)

# Top Positive Reviews
st.subheader(
    "Top Positive Reviews"
)

positive_reviews = df[
    df["predicted_sentiment"] == "Positive"
]

st.dataframe(
    positive_reviews[
        ["review_text"]
    ].head(10)
)

# Top Negative Reviews
st.subheader(
    "Top Negative Reviews"
)

negative_reviews = df[
    df["predicted_sentiment"] == "Negative"
]

st.dataframe(
    negative_reviews[
        ["review_text"]
    ].head(10)
)

#Review Length By Sentiment
df["review_length"] = (
    df["review_text"]
    .astype(str)
    .apply(len)
)

st.subheader(
    "Review Length by Sentiment"
)

box_fig = px.box(
    df,
    x="predicted_sentiment",
    y="review_length"
)

st.plotly_chart(
    box_fig,
    use_container_width=True
)

# Download Complete DataBase
st.subheader(
    "Export Data"
)
if not df.empty:
    csv = df.to_csv(
        index=False
    )
    st.download_button(
        label="📥 Download All Predictions",
        data=csv,
        file_name="customer_sentiment_data.csv",
        mime="text/csv"
    )
else:
    st.warning(
        "No data available for export."
    )


#Sidebar Navigation
st.sidebar.title(
    "📊 Dashboard Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "Analytics",
        "Database"
    ]
)

#Model Information
st.sidebar.markdown(
    """
    ### Model Information

    Model: Linear SVC

    Vectorizer: TF-IDF

    Accuracy: 88%

    Dataset: Amazon Reviews

    Classes:
    - Positive
    - Negative
    - Neutral
    """
)

# Footer

st.markdown("---")

st.markdown(
    """
    Developed using:

    - Python
    - Scikit-Learn
    - TF-IDF
    - Linear SVC
    - SQLite
    - FastAPI
    - Streamlit
    """
)