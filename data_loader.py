import json
import pandas as pd
import re
import io
from typing import Union

def load_reviews(json_source: Union[str, bytes, io.IOBase]) -> pd.DataFrame:
    """
    Load and flatten JSON review data into a DataFrame.
    Supports both file paths and uploaded file-like objects.
    """
    if isinstance(json_source, (str, bytes)):
        with open(json_source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        # Assume it's a file-like object (e.g., from Streamlit)
        string_io = io.StringIO(json_source.read().decode('utf-8'))
        raw_data = json.load(string_io)

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
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # remove special chars except punctuation
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
