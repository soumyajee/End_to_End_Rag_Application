import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
WEATHER_API_KEY ='7b76586d2f0b7f32e2d9b76f7053be71' 

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Graph State
class GraphState(TypedDict):
    input: str
    result: str
    output: str


# Router Logic
def router(state: GraphState):
    user_input = state["input"]
    if "weather" in user_input.lower():
        return {"result": "weather"}
    else:
        return {"result": "pdf"}


# Fetch Weather Information
def fetch_weather(state: GraphState):
    user_input = state["input"]
    city = "Unknown"
    if "in" in user_input:
        city = user_input.split("in")[-1].strip().split()[0]
    else:
        return {"output": "Please specify a city to get the weather information."}

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return {"output": f"The weather in {city} is {weather} with temperature {temp}°C."}
    else:
        return {"output": f"Failed to fetch weather info for {city}."}


# Fetch PDF Answer
def fetch_pdf_answer(state: GraphState):
    user_input = state["input"]
    pdf_path = "temp.pdf"  # Your PDF Path

    if not os.path.exists(pdf_path):
        return {"output": "PDF File Not Found"}

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create Vector Store
    vectorstore = Qdrant.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,
        api_key=QDRANT_API_KEY,
        collection_name="pdf_data"
    )

    # Search Relevant Info
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(user_input)

    if relevant_docs:
        answer = relevant_docs[0].page_content
    else:
        answer = "No relevant information found in PDF."

    return {"output": answer}
# Building LangGraph
graph = StateGraph(GraphState)

graph.add_node("router", RunnableLambda(router))
graph.add_node("weather", RunnableLambda(fetch_weather))
graph.add_node("pdf", RunnableLambda(fetch_pdf_answer))

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda x: x["result"],
    {
        "weather": "weather",
        "pdf": "pdf"
    }
)

graph.add_edge("weather", END)
graph.add_edge("pdf", END)

app = graph.compile()

# Example Inputs
input1 = {"input": "Tell me the weather in New York today"}
input2 = {"input": "Find about LLM Model in my PDF"}

result1 = app.invoke(input1)
result2 = app.invoke(input2)

print("Weather Result →", result1)
print("PDF Result     →", result2)
