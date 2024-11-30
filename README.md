# MedLlama
An AI-powered medical chatbot that provides personalized, accurate, and concise medical advice using FAISS proximity search and the Meta LLaMA language model.

Overview
MedLlama is a medical chatbot designed to improve accessibility to reliable healthcare information. It combines state-of-the-art AI technologies to deliver personalized responses to medical queries.

Key Highlights:

Personalized Medical Advice: Tailored recommendations based on user queries and relevant medical cases.
Proximity Search: Uses FAISS to retrieve the most relevant medical cases efficiently.
AI-Powered Language Model: Leverages Meta LLaMA for generating accurate and human-like responses.
Features
Natural Language Processing: Understands queries in everyday language.
FAISS Proximity Search: Quickly finds the most relevant cases from a medical database.
Personalized Responses: Combines user queries with retrieved cases to generate tailored advice.
Criticality Analysis: Identifies urgent cases and provides emergency alerts.
Feedback Mechanism: Collects user feedback to improve response accuracy.
Interactive UI: Built with Streamlit for an intuitive user experience.

MedLlama_Chatbot/
├── app.py                   # Main application script
├── final_embeddings.csv     # Medical case embeddings
├── faiss_index.idx          # FAISS index file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .env                     # Environment variables
└── feedback/                # MongoDB feedback data (if applicable)


Technologies Used
Programming Language: Python 3.9+
Framework: Streamlit (for the user interface)
AI Models:
SentenceTransformer (all-MiniLM-L6-v2) for embedding generation.
Meta LLaMA (meta-llama-3.1-8b-instruct) for natural language understanding.
Proximity Search: FAISS (for efficient case retrieval).
Database: MongoDB (for storing feedback).
Other Tools:
Python-dotenv (for managing environment variables).
Aiohttp (for asynchronous API requests).
Setup Instructions
Prerequisites
Python 3.9 or higher installed on your system.
MongoDB database for storing feedback (optional).
LM Studio installed and running with the Meta LLaMA model.


Demo
Example Query: "I have a headache. What should I do?"
Chatbot Response:
"Try resting in a dark, quiet room and drinking plenty of water. If the pain persists or worsens, consider consulting a healthcare professional."
Relevant Cases:
Rank 1: Patient with similar symptoms advised hydration and rest.
Rank 2: Severe headache with nausea diagnosed as a migraine.


Future Enhancements
Expand Medical Case Database:
Include more cases across diverse medical conditions.
Multi-Language Support:
Enable users to query in multiple languages.
Audio Input:
Allow voice-based queries for enhanced accessibility.
Real-Time Analytics:
Provide insights into user behavior and feedback trends.
Integration with Wearable Devices:
Incorporate real-time health data for personalized recommendations.
