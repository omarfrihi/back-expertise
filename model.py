import os
import joblib
from openai import OpenAI

# Load trained vectorizer and model
vectorizer = joblib.load("model/vectorizer.joblib")
model = joblib.load("model/classifier.joblib")

# Initialize OpenAI client
client = OpenAI(api_key="")

def chatgpt_justification(text: str):
    """
    Returns a tuple (found: bool, justification: str) using ChatGPT
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5.2-chat-latest",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. include in response true and false"},
                {"role": "user", "content": f"Verify this claim and provide a justification: {text}"}
            ]
        )
        answer = response.choices[0].message.content.strip()
        # Consider found if answer contains TRUE or FALSE
        found = "TRUE" in answer.upper() or "FALSE" in answer.upper()
        return found, answer
    except Exception as e:
        return False, f"ChatGPT API error: {str(e)}"

def fact_check(text: str):
    """
    Returns a dictionary with verdict, confidence, and ChatGPT justification
    """
    # Predict fake/real
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    
    fake_prob = probs[0]  # label 0
    real_prob = probs[1]  # label 1

    confidence = max(fake_prob, real_prob)

    if confidence < 0.6:
        verdict = "UNCERTAIN"
    elif real_prob > fake_prob:
        verdict = "TRUE"
    else:
        verdict = "FALSE"

    # Get ChatGPT justification
    found, justification = chatgpt_justification(text)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "chatgpt_found": found,
        "justification": justification
    }
