# Updated imports based on deprecation warnings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
from dotenv import load_dotenv
import os
import requests
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (ensure .env file exists)
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEATHER_API_KEY = '7b76586d2f0b7f32e2d9b76f7053be71'  # Ensure this is in your .env
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Initialize OpenAI LLM & Embeddings
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Weather API Logic
def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"Weather in {city}: {data['weather'][0]['description']}, Temp: {data['main']['temp']}°C"
    return "City not found or invalid query!"

# PDF Query using LangChain + Qdrant
def fetch_pdf_answer(query, pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,   # Keep it False unless using grpc
        api_key=QDRANT_API_KEY,
        collection_name="pdf_data"
    )

    retriever = vectorstore.as_retriever()
    result = retriever.get_relevant_documents(query)
    if result:
        return result[0].page_content
    return "No relevant info found in PDF."

# Decision Node to decide between Weather or PDF fetching
class DecisionNode:
    def __init__(self):
        pass

    def decide(self, query, pdf_path=None):
        normalized_query = query.lower()

        if not pdf_path and normalized_query:
            return fetch_weather(query)  # Call weather API
        elif pdf_path:
            return fetch_pdf_answer(query, pdf_path)  # Fetch answer from PDF
        else:
            return "Invalid input or no file uploaded."
