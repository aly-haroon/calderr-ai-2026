import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the API key from .env file
load_dotenv()

# Connect to Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}")
])

# Build the chain
chain = prompt | llm | StrOutputParser()

# Run it!
response = chain.invoke({"question": "What is an AI agent and why does it matter?"})
print(response)