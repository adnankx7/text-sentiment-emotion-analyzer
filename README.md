# 🧠 Text Sentiment & Emotion Analyzer

A simple Streamlit web application that uses pre-trained NLP models to analyze **sentiment** and **emotion** in user-inputted text. Built using Hugging Face Transformers and Streamlit.

---

## 🔍 Features

- **Sentiment Analysis** using [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- **Emotion Classification** using [`bhadresh-savani/distilbert-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
- Clean and interactive UI with **Streamlit**
- Confidence score for sentiment predictions

---

## 📦 Dependencies

- Python 3.10+
- [Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [Torch](https://pytorch.org/)

Install required packages with:

```bash
pip install -r requirements.txt

## 🚀 How to Run

Clone the repository:

   ```bash
   git clone https://github.com/yourusername/text-analyzer-app.git
   cd text-analyzer-app
   ```bash
   streamlit run app.py
