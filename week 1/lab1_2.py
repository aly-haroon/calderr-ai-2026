import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

# Connect to Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Mock database of facts
FACTS_DATABASE = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "langchain": "LangChain is a framework for building applications powered by large language models.",
    "groq": "Groq is a company that provides ultra-fast AI inference hardware and API.",
    "pakistan": "Pakistan is a country in South Asia with a population of over 220 million people.",
    "calderr": "CalderR is an AI company focused on rethinking how work works using agentic AI.",
}

# Tool 1: Calculator
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Calculator result: {expression} = {result}"
    except:
        return "Error: Could not calculate that expression"

# Tool 2: Search database
def search(query: str) -> str:
    query_lower = query.lower()
    for key, value in FACTS_DATABASE.items():
        if key in query_lower:
            return f"Search result: {value}"
    return "Search result: No information found in database"

# Tool 3: General Groq answer
def general_answer(question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer briefly in 2-3 sentences."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# The BRAIN — decides which tool to use
def decide_tool(user_question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI agent. Based on the user's question, decide which tool to use.
        
Available tools:
- calculator: for any math calculations or number problems
- search: for questions about python, langchain, groq, pakistan, or calderr
- general: for everything else

Respond with ONLY one word: calculator, search, or general"""),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": user_question}).strip().lower()

# Main ReAct loop
console.print(Panel("ReAct Agent — I can calculate, search, or answer anything!", 
                     style="bold blue"))

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input == "/exit":
        console.print("[bold red]Goodbye![/bold red]")
        break
    
    if not user_input:
        continue

    # THINK — decide which tool
    console.print("[dim]Agent thinking...[/dim]")
    tool = decide_tool(user_input)
    console.print(f"[bold yellow]Agent decided to use: {tool}[/bold yellow]")

    # ACT — use the tool
    if tool == "calculator":
        # Extract the math expression
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract only the math expression from the question. Return only the expression, nothing else."),
            ("user", "{question}")
        ])
        extract_chain = extract_prompt | llm | StrOutputParser()
        expression = extract_chain.invoke({"question": user_input})
        result = calculator(expression)
    elif tool == "search":
        result = search(user_input)
    else:
        result = general_answer(user_input)

    # OBSERVE + ANSWER
    console.print(f"\n[bold green]Agent:[/bold green] {result}")