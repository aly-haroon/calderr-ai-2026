import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel

# Setup
load_dotenv()
console = Console()

# Connect to Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template with conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant named CalderBot."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

chain = prompt | llm

# Store conversation history
history = []

console.print(Panel("Welcome to CalderBot! Type /clear to reset or /exit to quit.", 
                     style="bold blue"))

# Main chat loop
while True:
    user_input = input("\nYou: ").strip()
    
    if user_input == "/exit":
        console.print("[bold red]Goodbye![/bold red]")
        break
    
    if user_input == "/clear":
        history = []
        console.print("[bold yellow]Conversation cleared![/bold yellow]")
        continue
    
    if not user_input:
        continue
    
    # Get response from Groq
    response = chain.invoke({
        "history": history,
        "question": user_input
    })
    
    bot_reply = response.content
    
    # Save to history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=bot_reply))
    
    # Display response
    console.print(f"\n[bold green]CalderBot:[/bold green] {bot_reply}")
    console.print(f"[dim]Messages in history: {len(history)}[/dim]")