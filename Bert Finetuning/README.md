# Anxiety Detection and Sentiment Analysis App

This application consists of a FastAPI backend for anxiety detection and sentiment analysis, and a Streamlit frontend for user interaction.

## Setup

### 1. Create a Python Virtual Environment

For Windows:
```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

### 2. Install Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the FastAPI Backend

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

You can access the API documentation at `http://localhost:8000/docs`

### 2. Start the Streamlit Frontend

In a new terminal window, run:
```bash
streamlit run streamlit_app.py
```

The Streamlit app will open automatically in your default web browser, typically at `http://localhost:8501`

## Using the Application

1. Enter your text in the text area
2. Choose either:
   - "Detect Anxiety" to analyze anxiety levels in the text
   - "Analyze Sentiment" to get sentiment scores for the text
3. Results will be displayed below the buttons

## API Endpoints

- POST `/predict`: Anxiety detection
- POST `/sentiment`: Sentiment analysis

Both endpoints accept JSON with a "text" field:
```json
{
    "text": "Your text here"
}
```

## Model Information

The application uses:
- A fine-tuned BERT model for anxiety detection
- VADER Sentiment for sentiment analysis

## Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment:
```bash
deactivate
```