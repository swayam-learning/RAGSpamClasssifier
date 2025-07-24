# Spam Classification and Explaination Chatbot

This project implements a two-step email spam detection system that combines fast classification with explainable reasoning. The goal is not just to identify whether an email is spam, but to also provide a natural language explanation grounded in historical email data.

## Key Components

- **DistilBERT** for fast binary spam classification  
- **FAISS** and **SentenceTransformer** to retrieve similar past emails  
- **Mistral 7B** to generate contextual explanations  

## Motivation

This system was built to explore how retrieval-augmented generation (RAG) can be used for interpretable machine learning, particularly in scenarios like spam filtering where understanding the rationale behind a decision is critical.

## Features

- High-speed classification with lightweight transformer  
- Context-aware reasoning using real email examples  
- Modular and extendable pipeline for research or deployment  

---

*For academic or research purposes.*
