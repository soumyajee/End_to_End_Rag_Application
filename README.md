# End_to_End_Rag_Application
AI Agentic Pipeline using LangChain, LangGraph, LangSmith, and Streamlit
Overview

This project demonstrates an AI pipeline using:

    LangChain

    LangGraph

    LangSmith

    Vector Databases (Qdrant)

    Streamlit UI

Features:

    Fetch real-time weather data from OpenWeatherMap API.

    Answer user queries from a PDF using RAG (Retrieval Augmented Generation).

    Decision-making node (Weather API vs PDF RAG Search).

    Store embeddings in Qdrant.

    LangSmith testing and tracing.

    Streamlit UI for interaction.

    Unit tests for API, LLM logic, and pipeline flow.

Project Structure

├── app.py                → Main Application Logic (LangGraph Pipeline)
├── test_langsmith.py     → LangSmith evaluation & tracing
├── Langraph_testing.py      → LangGraph specific node testing
├── Testing.py               → Unit test cases for API, LLM, and retrieval
├── requirements.txt      → Python dependencies
├── README.md             → Project documentation (this file)
└── sample.pdf            → PDF for RAG-based QA

Setup Instructions
1. Clone the Repository

git clone <your-repo-url>
cd <your-project-directory>

2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3. Install Dependencies

pip install -r requirements.txt

Run the Application

streamlit run app.py

Run Tests

python test_langsmith.py
python  Langraph_testing.py
python testing_seperate_cases.py 


How it Works
File	Purpose
app.py	Implements LangGraph flow - decides between Weather API or PDF QA
langraph_test.py	Tests specific LangGraph nodes and their logic
langsmith_test.py	Tests tracing and evaluation of responses using LangSmith
test.py	API tests, LangChain test cases, retrieval logic tests
Tech Stack Used

    LangChain

    LangGraph

    LangSmith

    Qdrant (Vector DB)

    OpenWeatherMap API

    Streamlit (UI)

    Pytest (Testing)

Environment Variables

Create a .env file in your root directory and add:

OPENWEATHER_API_KEY = your_openweather_api_key
LANGCHAIN_API_KEY = your_langchain_api_key
QDRANT_URL = your_qdrant_instance_url
QDRANT_API_KEY=your_qdrant_api_key
LANGSMITH_API_KEY=langsmith_api_key
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT=langsmith_api_url
