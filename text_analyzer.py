import transformers 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class TextAnalyzer:
    def __init__(self):
        
        ## Initialize Sentiment Model & Pipeline
        sentiment_model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        self.sentiment_pipeline = pipeline('sentiment-analysis', 
                                           model = self.sentiment_model,
                                           tokenizer = self.sentiment_tokenizer)
        
        self.sentiment_labels = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }
                
        
        ## Initialize Emotion Model & Pipeline 
        emotion_model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
        self.emotion_pipeline = pipeline('text-classification',
                                         model = self.emotion_model, 
                                        tokenizer = self.emotion_tokenizer)
        
    
    def sentiment_analyze(self, text):
        result = self.sentiment_pipeline(text)[0]
        return {
                'label' : self.sentiment_labels.get(result['label'], result['label']),
                'score': round(result['score'], 4)
        }
    
    def emotion_analyze(self,text):
        result = self.emotion_pipeline(text)[0]
        return result['label']
    
