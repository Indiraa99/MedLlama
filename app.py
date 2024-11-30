import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Tuple, List
import os
from pymongo import MongoClient
import faiss
import logging
import asyncio
import aiohttp
import re

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize MongoDB connection
def get_mongo_client():
    mongo_uri = os.getenv("MONGODB_URI")
    return MongoClient(mongo_uri)

def store_feedback_to_mongo(query, response, categories, feedback):
    """Store feedback in MongoDB."""
    try:
        client = get_mongo_client()
        db = client['medical_rag_bot']
        collection = db['feedback']

        feedback_data = {
            "query": query,
            "response": response,
            "categories": categories,
            "feedback": feedback,
            "timestamp": datetime.now()
        }
        collection.insert_one(feedback_data)
        logging.info("Feedback saved to MongoDB.")
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
    finally:
        client.close()

# Constants
EMERGENCY_KEYWORDS = {
    'severe': 0.8, 'emergency': 1.0, 'critical': 0.9, 'urgent': 0.8, 'life-threatening': 1.0,
    'unconscious': 1.0, 'bleeding heavily': 0.9, 'difficulty breathing': 0.9, 'heart attack': 1.0, 'stroke': 1.0
}

SYMPTOM_CATEGORIES = {
    'respiratory': ['breathing', 'cough', 'wheeze', 'chest congestion'],
    'cardiac': ['chest pain', 'heart', 'palpitations', 'shortness of breath'],
    'neurological': ['headache', 'dizziness', 'numbness', 'seizure'],
    'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],
    'musculoskeletal': ['joint pain', 'muscle pain', 'back pain', 'injury'],
    'psychological': ['anxiety', 'depression', 'stress', 'panic'],
}

# Criticality Analyzer
class CriticalityAnalyzer:
    def __init__(self):
        self.emergency_keywords = EMERGENCY_KEYWORDS
        self.symptom_categories = SYMPTOM_CATEGORIES

    def analyze_criticality(self, query: str) -> Tuple[float, str, List[str]]:
        query_lower = query.lower()
        score = 0
        matched_keywords = []

        # Analyze criticality based on keywords
        for keyword, weight in self.emergency_keywords.items():
            if keyword in query_lower:
                score = max(score, weight)
                matched_keywords.append(keyword)

        if score >= 0.9:
            urgency_level = "🚨 EMERGENCY ALERT 🚨 - Call emergency services immediately!"
        elif score >= 0.7:
            urgency_level = "⚠️ URGENT CARE NEEDED ⚠️ - Contact your healthcare provider."
        elif score >= 0.4:
            urgency_level = "MODERATE - Schedule a medical appointment."
        else:
            urgency_level = "LOW - Monitor symptoms and practice self-care."

        # Categorize symptoms
        relevant_categories = [
            category for category, symptoms in self.symptom_categories.items()
            if any(symptom in query_lower for symptom in symptoms)
        ]

        return score, urgency_level, relevant_categories

@st.cache_resource
def load_embedding_model():
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index_and_data(faiss_index_path, data_path):
    """Load FAISS index and embedding data."""
    df = pd.read_csv(data_path)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x), dtype=np.float32))
    d = df["embeddings"].iloc[0].shape[0]
    index = faiss.IndexFlatL2(d)
    index.add(np.vstack(df["embeddings"].values))
    return index, df

def search_faiss(query_embedding, index, df, top_k=1):
    """Search FAISS for relevant cases."""
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(df):
            results.append({
                "rank": i + 1,
                "description": df.iloc[idx]["Description"],
                "patient": df.iloc[idx]["Patient"],
                "doctor_response": df.iloc[idx]["Doctor"],
                "distance": distances[0][i],
            })
    return results

async def generate_llm_response(query, faiss_results, lmstudio_url, model_name):
    """Generate response using LM Studio."""
    context_items = [
        f"Description: {result['description']} Patient: {result['patient']} Doctor: {result['doctor_response']}"
        for result in faiss_results
    ]
    context = "No relevant cases found." if not faiss_results else " ".join(context_items)

    system_prompt = (
    "You are a medical assistant. Provide accurate, complete, and concise medical advice "
    "based only on the user's query and relevant retrieved cases. Ensure that your response "
    "is fully formed and does not appear truncated."
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Relevant cases:\n{context}\n\nUser query: {query}"}
    ]

    payload = {"model": model_name, "messages": messages, "max_tokens": 500, "temperature": 0.7}

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{lmstudio_url}/v1/chat/completions", json=payload) as response:
            return (await response.json())["choices"][0]["message"]["content"].strip()

def chatbot_pipeline(query, index, df, embedding_model):
    """Complete chatbot pipeline: Retrieve FAISS results and generate response."""
    query_embedding = embedding_model.encode(query).reshape(1, -1).astype(np.float32)
    faiss_results = search_faiss(query_embedding, index, df, top_k=3)
    lmstudio_url = "http://localhost:1234"
    model_name = "meta-llama-3.1-8b-instruct"
    response = asyncio.run(generate_llm_response(query, faiss_results, lmstudio_url, model_name))
    return response, faiss_results

# Feedback Section
def collect_feedback(query, response, categories):
    """Collect user feedback on the response and store it in MongoDB."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Helpful"):
            st.success("Thank you for your feedback!")
            store_feedback_to_mongo(query, response, categories, "Helpful")
            
    with col2:
        if st.button("👎 Not Helpful"):
            st.error("We'll work on improving our responses.")
            store_feedback_to_mongo(query, response, categories, "Not Helpful")
            
    with col3:
        if st.button("⚕️ Need Professional Help"):
            st.warning("Please consult a healthcare professional.")
            store_feedback_to_mongo(query, response, categories, "Professional Help Needed")

# Streamlit UI
st.title("MedLlama - Enhanced Medical Chatbot")
st.write("This chatbot uses FAISS for proximity search and Meta LLaMA for generating responses.")

# Load resources
embedding_model = load_embedding_model()
faiss_index_path = "C:\\HAP774\\New folder (2)\\MedLlama_Chatbot\\faiss_index.idx"
data_path = "C:\\HAP774\\New folder (2)\\MedLlama_Chatbot\\final_embeddings.csv"
index, df = load_faiss_index_and_data(faiss_index_path, data_path)
analyzer = CriticalityAnalyzer()

# User input
query = st.text_input("Enter your medical query:")

if st.button("Get Response"):
    if query.strip():
        # Analyze criticality
        criticality_score, urgency_level, categories = analyzer.analyze_criticality(query)
        if criticality_score >= 0.9:
            st.error(urgency_level)
        elif criticality_score >= 0.7:
            st.warning(urgency_level)
        else:
            st.info(urgency_level)

        # Run chatbot pipeline
        response, results = chatbot_pipeline(query, index, df, embedding_model)

        # Display chatbot response
        st.subheader("Chatbot Response:")
        st.write(response)

        # Display related categories
        st.subheader("Related Medical Categories:")
        for category in categories:
            st.write(f"- {category.capitalize()}")

        # Display relevant cases
        st.subheader("Relevant Cases:")
        for result in results:
            st.write(f"**Rank {result['rank']}**")
            st.write(f"**Description:** {result['description']}")
            st.write(f"**Patient's Statement:** {result['patient']}")
            st.write(f"**Doctor's Response:** {result['doctor_response']}")
            st.write(f"**Distance:** {result['distance']}")
            st.write("---")

        # Collect feedback
        st.write("### Feedback:")
        collect_feedback(query, response, categories)
