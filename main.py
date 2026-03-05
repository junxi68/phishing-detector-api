from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

app = FastAPI()

# Load model at startup
model = XLMRobertaForSequenceClassification.from_pretrained(
    "JunXi888/phishing-detector"
)
tokenizer = XLMRobertaTokenizer.from_pretrained(
    "JunXi888/phishing-detector"
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs.max().item()
        prediction = torch.argmax(probs).item()

    label = "phishing" if prediction == 1 else "legitimate"

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }