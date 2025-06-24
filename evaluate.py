from data_loader import load_reviews, preprocess_reviews
from nlp_service import analyze_sentiment, classify_topics
import pandas as pd
from tqdm import tqdm


CANDIDATE_LABELS = [
    "Teaching Quality",
    "Grading",
    "Workload",
    "Class Engagement",
    "Lecture Clarity",
    "Professor Personality"
]

def run_evaluation(json_path: str = "all_reviews.json", output_csv: str = "evaluated_reviews.csv"):
    # Load and clean the data
    df = load_reviews(json_path)
    df = preprocess_reviews(df)

    # Prepare columns for results
    sentiments = []
    scores = []
    predicted_topics = []

    tqdm.pandas(desc="Analyzing reviews")

    for comment in tqdm(df['cleaned_comment'], desc="Processing", total=len(df)):
        # Sentiment
        sentiment_result = analyze_sentiment(comment)
        sentiments.append(sentiment_result['sentiment'])
        scores.append(sentiment_result['score'])

        # Topic Classification
        topic_result = classify_topics(comment, CANDIDATE_LABELS)
        predicted_topics.append(topic_result['predicted_label'])

    df['predicted_sentiment'] = sentiments
    df['sentiment_score'] = scores
    df['predicted_topic'] = predicted_topics

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    return df


if __name__ == "__main__":
    run_evaluation()
