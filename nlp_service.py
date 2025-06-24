from transformers import pipeline
from typing import List, Dict

# Load Hugging Face pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_sentiment(text: str) -> Dict:
    """
    Return sentiment label and score.
    Example: {'label': 'POSITIVE', 'score': 0.998}
    """
    result = sentiment_analyzer(text)[0]
    return {"sentiment": result['label'], "score": result['score']}


def classify_topics(text: str, candidate_labels: List[str]) -> Dict:
    """
    Return best-matching topic label and all scores.
    """
    result = zero_shot_classifier(text, candidate_labels)
    return {
        "predicted_label": result['labels'][0],
        "scores": dict(zip(result['labels'], result['scores']))
    }


if __name__ == "__main__":
    sample_text = "The professor was very engaging and gave useful feedback on essays."
    sentiment = analyze_sentiment(sample_text)
    topics = classify_topics(sample_text, ["Teaching Quality", "Grading", "Course Load", "Class Engagement"])
    
    print("Sentiment:", sentiment)
    print("Topics:", topics)
