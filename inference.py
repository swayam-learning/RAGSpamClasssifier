# inference.py

import os
import torch
import faiss
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Load .env only for local testing (safe for dev, ignored in production)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

# Load token securely from environment
mistral_token = os.getenv("HF_TOKEN")
if mistral_token is None:
    raise ValueError("Hugging Face token not found in environment. Please set HF_TOKEN.")

# Model IDs
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
classifier_model_id = "distilbert-base-uncased-finetuned-sst-2-english"

# Load Mistral model and tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id, token=mistral_token)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=mistral_token
)

# Load classifier
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_id).to("cuda")

# Load email dataset
csv_path = "email_spam.csv"
try:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
except FileNotFoundError:
    df = pd.DataFrame({
        'type': ['ham', 'spam'],
        'text': ['Hi, how are you?', 'Click here to win a free prize!']
    })

assert 'text' in df.columns and 'type' in df.columns, "CSV must contain 'text' and 'type' columns"

documents = df['text'].fillna("").tolist()
labels = df['type'].fillna("").tolist()

# Create sentence embeddings and FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(documents, convert_to_tensor=True).cpu().numpy().astype("float32")

index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# Retrieval
def retrieve_docs(query, k=3):
    query_embedding = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, k)
    return [f"[{labels[i].upper()}] {documents[i][:350]}..." for i in indices[0]]

# Fast classifier
def classify_spam_fast(email_text):
    inputs = classifier_tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = classifier_model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    confidence = logits.softmax(dim=1)[0][predicted_class_id].item()
    verdict = "SPAM" if predicted_class_id == 0 else "NOT SPAM"
    return verdict, confidence

# Reasoning generator
def generate_reasoning(query, verdict, k=3):
    context_docs = retrieve_docs(query, k)
    context = "\n\n".join(context_docs)

    prompt = f"""<s>[INST] An email has been classified as **{verdict}**.
Based on the context of similar past emails provided below, give a brief, one-paragraph explanation for this classification.
### Context:
{context}
### New Email:
"{query}"
[/INST] Reasoning:"""

    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    outputs = mistral_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=mistral_tokenizer.eos_token_id
    )

    input_length = inputs['input_ids'].shape[1]
    result = mistral_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    return result.strip()

# Entry point for Hugging Face Inference Endpoint
def pipeline(payload: dict):
    query = payload.get("text", "").strip()
    if not query:
        return {"error": "No input text provided."}

    verdict, confidence = classify_spam_fast(query)
    reasoning = generate_reasoning(query, verdict)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "reasoning": reasoning
    }
