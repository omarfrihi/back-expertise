import os
import joblib
from openai import OpenAI

# =====================================================
# 1. Load trained ML pipeline (vectorizer + model)
# =====================================================
pipeline = joblib.load("model/fake_news_pipeline.joblib")

# =====================================================
# 2. Initialize OpenAI client (API key from env)
# =====================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================
# 3. ChatGPT justification (structured output)
# =====================================================
def chatgpt_justification(text: str):
    """
    Returns a tuple (verdict: str, justification: str)
    Verdict is one of: TRUE, FALSE, UNCERTAIN
    """
    try:
        response = client.responses.create(
            model="gpt-5.2-chat-latest",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking assistant. "
                        "Respond STRICTLY in this format:\n\n"
                        "VERDICT: TRUE | FALSE | UNCERTAIN\n"
                        "JUSTIFICATION: <short explanation>"
                    )
                },
                {
                    "role": "user",
                    "content": f"Verify the following claim:\n{text}"
                }
            ],
        )

        answer = response.output_text.strip()

        # Parse verdict safely
        if "VERDICT: TRUE" in answer:
            verdict = "TRUE"
        elif "VERDICT: FALSE" in answer:
            verdict = "FALSE"
        else:
            verdict = "UNCERTAIN"

        return verdict, answer

    except Exception as e:
        return "UNCERTAIN", f"OpenAI API error: {str(e)}"

# =====================================================
# 4. Main fact-checking function
# =====================================================
def fact_check(text: str):
    """
    Returns ML verdict + confidence + LLM justification
    """
    # ---- ML prediction ----
    probs = pipeline.predict_proba([text])[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    confidence = float(max(fake_prob, real_prob))

    if confidence < 0.6:
        ml_verdict = "UNCERTAIN"
    elif real_prob > fake_prob:
        ml_verdict = "TRUE"
    else:
        ml_verdict = "FALSE"

    # ---- LLM justification (only if useful) ----
    llm_verdict, justification = chatgpt_justification(text)

    return {
        "verdict": ml_verdict,
        "confidence": round(confidence, 2),
        "llm_verdict": llm_verdict,
        "justification": justification
    }

