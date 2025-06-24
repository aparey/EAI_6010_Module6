import json
import pandas as pd
import re

def load_reviews(json_path: str = "all_reviews.json") -> pd.DataFrame:
    """Load and flatten JSON review data into a DataFrame."""
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Flatten all nested lists into one list of reviews
    all_reviews = []
    for professor_reviews in raw_data:
        all_reviews.extend(professor_reviews)

    df = pd.DataFrame(all_reviews)
    return df


def clean_comment(text: str) -> str:
    """Clean and normalize comment text."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # remove special chars except punct
    return text.strip()


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the Comment column and drop rows with empty comments."""
    df = df.copy()
    df = df[df['Comment'].notna()]  # drop rows with missing Comment
    df['cleaned_comment'] = df['Comment'].apply(clean_comment)
    return df


if __name__ == "__main__":
    df = load_reviews("all_reviews.json")
    df_clean = preprocess_reviews(df)
    print(df_clean[['professor', 'course_id', 'Quality', 'Difficulty', 'cleaned_comment']].head())
