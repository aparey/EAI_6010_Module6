import streamlit as st
import pandas as pd
from data_loader import load_reviews, preprocess_reviews
from nlp_service import analyze_sentiment, classify_topics

# Define candidate topics
CANDIDATE_LABELS = [
    "Teaching Quality", "Grading", "Workload", 
    "Class Engagement", "Lecture Clarity", "Professor Personality"
]

st.set_page_config(page_title="Student Review Analyzer", layout="wide")
st.title("📊 Student Review Analyzer")
st.write("Analyze sentiment and detect topics from student feedback using NLP.")

# File uploader
uploaded_file = st.file_uploader("Upload the all_reviews.json file", type="json")

if uploaded_file:
    with st.spinner("Loading and cleaning reviews..."):
        df_raw = load_reviews(uploaded_file)
        df = preprocess_reviews(df_raw)

    st.success(f"Loaded {len(df)} reviews.")
    st.dataframe(df[['professor', 'course_id', 'Comment']].head())

    # Select review for analysis
    selected_index = st.selectbox("Pick a review to analyze", range(len(df)), format_func=lambda i: df.iloc[i]['Comment'][:100])
    selected_comment = df.iloc[selected_index]['cleaned_comment']

    st.subheader("🔍 NLP Results")
    st.write(f"**Selected Comment:** {selected_comment}")

    sentiment_result = analyze_sentiment(selected_comment)
    topic_result = classify_topics(selected_comment, CANDIDATE_LABELS)

    st.markdown(f"**Sentiment:** `{sentiment_result['sentiment']}` (Confidence: `{sentiment_result['score']:.2f}`)")
    st.markdown(f"**Predicted Topic:** `{topic_result['predicted_label']}`")

    with st.expander("🔎 View topic scores"):
        scores_df = pd.DataFrame.from_dict(topic_result['scores'], orient='index', columns=["Confidence"]).reset_index()
        scores_df.columns = ["Topic", "Confidence"]
        st.dataframe(scores_df)

    # Full evaluation
    if st.button("Run Full Evaluation on Dataset"):
        with st.spinner("Running NLP analysis on all comments..."):
            from evaluate import run_evaluation
            df_eval = run_evaluation(json_path=uploaded_file)
        st.success("Evaluation complete!")
        st.dataframe(df_eval[['cleaned_comment', 'predicted_sentiment', 'predicted_topic']].head())
        st.download_button("Download Full CSV", data=df_eval.to_csv(index=False), file_name="evaluated_reviews.csv", mime="text/csv")
