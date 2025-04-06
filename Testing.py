from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
from dotenv import load_dotenv
import os
import requests
from langsmith_Testing import Test
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (ensure .env file exists)
from dotenv import load_dotenv
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
    city = city.strip()  # Clean the city name of extra spaces
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data.get('weather', [{}])[0].get('description', 'No description available')
        temp = data.get('main', {}).get('temp', 'Temperature not available')
        return f"Weather in {city}: {weather}, Temp: {temp}°C"
    else:
        return "City not found or invalid query!"

# PDF Query using LangChain + Qdrant
def fetch_pdf_answer(query, pdf_path):
    logger.info(f"Received Query: {query}, Source: PDF")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,  # Keep it False unless using grpc
        api_key=QDRANT_API_KEY,
        collection_name="pdf_data"
    )

    retriever = vectorstore.as_retriever()
    result = retriever.get_relevant_documents(query)
    if result:
        logger.info("Relevant document found.")
        return result[0].page_content
    else:
        logger.warning("No relevant info found in PDF.")
        return "No relevant info found."

# Decision Node to decide between Weather or PDF fetching
class DecisionNode:
    def __init__(self):
        pass

    def decide(self, query, pdf_path=None):
        logger.info(f"Making a decision based on query: {query} and PDF path: {pdf_path}")

        # Normalize the query to lowercase for better matching
        normalized_query = query.lower()

        # If a city name is provided
        if not pdf_path and normalized_query:
            # We assume it's a city if it's a valid city name (we could validate further if needed)
            logger.info(f"City provided: {query}. Calling Weather API.")
            return fetch_weather(query)  # Fetch weather data for the city

        # If a PDF is provided
        elif pdf_path:
            logger.info(f"PDF provided: {pdf_path}. Fetching from PDF.")
            return fetch_pdf_answer(query, pdf_path)  # Fetch answer from the PDF

        # If neither city nor PDF path is provided, return an error message
        else:
            logger.warning("Invalid query or missing PDF file.")
            return "Invalid input or no file uploaded."

# Updated fetch_weather function with better error handling
def fetch_weather(city):
    city = city.strip()  # Clean the city name of extra spaces
    logger.info(f"Fetching weather for city: {city}")  # Log the city name being fetched
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data.get('weather', [{}])[0].get('description', 'No description available')
        temp = data.get('main', {}).get('temp', 'Temperature not available')
        return f"Weather in {city}: {weather}, Temp: {temp}°C"
    else:
        logger.error(f"Error fetching weather data for {city}: {response.status_code}")
        return "City not found or invalid query!"


