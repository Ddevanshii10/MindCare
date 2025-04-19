from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI()

# Enable CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, tokenizer, and label encoder
model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")
label_encoder = pickle.load(open('label_encoder1.pkl', 'rb'))

# Prepare stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Inference function
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs

# Pydantic model for request body
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_label(input: TextInput):
    print(input)
    result = detect_anxiety(input.text)
    return {"input": input.text, "predicted_label": result}

@app.post("/sentiment")
def predict_sentiment(input: TextInput):
    result = sentiment_analysis(input.text)
    return {"input": input.text, "sentiment_scores": result}

class TextInput(BaseModel):
    responses: list[str]

def calculate_risk_score(anxiety_class):
    high_risk = ['Suicidal', 'Bipolar']
    medium_risk = ['Depression', 'Anxiety', 'Stress', 'Personality disorder']
    low_risk = ['Normal']
    
    if anxiety_class in high_risk:
        return "HIGH"
    elif anxiety_class in medium_risk:
        return "MEDIUM"
    else:
        return "LOW"

@app.post("/analyze")
def analyze_responses(input: TextInput):
    results = []
    
    for text in input.responses:
        # Get anxiety prediction
        anxiety_result = detect_anxiety(text)
        
        # Get sentiment analysis
        sentiment_result = sentiment_analysis(text)
        
        # Calculate risk score
        risk_score = calculate_risk_score(anxiety_result)
        
        # Combine results for each response
        results.append({
            "text": text,
            "anxiety_level": anxiety_result,
            "risk_score": risk_score,
            "sentiment_scores": sentiment_result
        })
    
    # Calculate overall risk score based on the highest risk found
    risk_scores = [r["risk_score"] for r in results]
    overall_risk = "LOW"
    if "HIGH" in risk_scores:
        overall_risk = "HIGH"
    elif "MEDIUM" in risk_scores:
        overall_risk = "MEDIUM"
    
    return {
        "analysis_results": results,
        "summary": {
            "total_responses": len(results),
            "anxiety_levels": [r["anxiety_level"] for r in results],
            "overall_risk_score": overall_risk,
            "average_sentiment": {
                "positive": sum(r["sentiment_scores"]["pos"] for r in results) / len(results),
                "negative": sum(r["sentiment_scores"]["neg"] for r in results) / len(results),
                "neutral": sum(r["sentiment_scores"]["neu"] for r in results) / len(results),
                "compound": sum(r["sentiment_scores"]["compound"] for r in results) / len(results)
            }
        }
    }
