import streamlit as st
from text_analyzer import TextAnalyzer

st.set_page_config(page_title="Text Sentiment & Emotion Analyzer")
st.title("🧠 Text Sentiment & Emotion Analyzer")

@st.cache_resource
def load_analyzer():
    return TextAnalyzer()

analyzer = load_analyzer()

text =   st.text_area('Enter your text here: ',height=150)

# Button to analyze
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            sentiment = analyzer.sentiment_analyze(text)
            emotion = analyzer.emotion_analyze(text)

        # Display results
        st.subheader("Sentiment")
        st.write(f"**Label:** {sentiment['label']}")
        st.write(f"**Confidence:** {sentiment['score']}")

        st.subheader("Emotion")
        st.write(f"**Predicted Emotion:** {emotion}")