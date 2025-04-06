import os
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API KEYS
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
WEATHER_API_KEY = '7b76586d2f0b7f32e2d9b76f7053be71'  # move to .env ideally

# LangChain Models
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# Weather API with LangSmith tracing
@traceable(name="Weather API Call")
def fetch_weather(city):
    city = city.strip()
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data.get('weather', [{}])[0].get('description', 'No description')
        temp = data.get('main', {}).get('temp', 'Temp N/A')
        result = f"Weather in {city}: {weather}, Temp: {temp}°C"
        logger.info(result)
        return result
    else:
        logger.warning(f"City {city} not found!")
        return "City not found!"


# PDF QnA with LangSmith tracing
@traceable(name="PDF QnA Search")
def fetch_pdf_answer(query, pdf_path):
    if not os.path.exists(pdf_path):
        return "PDF file not found!"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,
        api_key=QDRANT_API_KEY,
        collection_name="pdf_data"
    )

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based on the context provided.

Context:
{context}

Question:
{question}
"""
    )

    qa_chain = LLMChain(llm=llm, prompt=prompt)

    answer = qa_chain.run({
        "context": context,
        "question": query
    })

    logger.info(f"Question: {query}")
    logger.info(f"Answer: {answer}")

    return answer
if __name__ == "__main__":
    # Weather Example
    city = "Hyderabad"
    weather = fetch_weather(city)
    logger.info(weather)   # Output to console via logs

    # PDF QnA Example
    query = "What are the different kinds of LLM models"
    pdf_path = "temp.pdf"

    answer = fetch_pdf_answer(query, pdf_path)
    logger.info(answer)   # Output to console via logs
