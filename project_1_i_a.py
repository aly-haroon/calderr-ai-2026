import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

# ── Setup ──────────────────────────────────────────────────────
load_dotenv()
console = Console()

# ── Domain Definition ──────────────────────────────────────────

DOMAIN = "Football"
SYSTEM_PROMPT = """You are an expert Football assistant named CalderBot.

Your expertise covers:
- Football history and records
- Current teams, players, and leagues
- Match tactics and formations
- Transfer news and rumors
- FIFA World Cup, Champions League, Premier League and all major competitions

Rules you must follow:
1. Only answer questions related to football
2. If asked about anything outside football, politely refuse and redirect
3. Give detailed, passionate answers like a true football expert
4. Back your answers with stats and facts when possible

You are part of the CalderR Agentic AI Internship 2026 program."""
# ── LLM Setup ──────────────────────────────────────────────────
# temperature=0.7 — balanced between creative and precise
# Good for technical conversations
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Prompt Template ────────────────────────────────────────────
# MessagesPlaceholder is key here — it inserts the entire
# conversation history into the prompt automatically
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

chain = prompt | llm

# ── Conversation History ────────────────────────────────────────
# This list stores every message in the conversation
# HumanMessage = what you said
# AIMessage = what the bot said
history = []

# ── Token Tracker ───────────────────────────────────────────────
# Tracks total tokens used across the entire session
total_tokens = 0

# ── Helper Functions ────────────────────────────────────────────
def display_welcome():
    """Shows the welcome screen when the app starts"""
    console.print(Rule(style="blue"))
    console.print(Panel(
        f"[bold blue]CalderBot[/bold blue] — Your {DOMAIN} Assistant\n"
        f"[dim]Commands: /clear (reset history) · /exit (quit) · /history (show history)[/dim]",
        style="blue"
    ))
    console.print(Rule(style="blue"))

def display_response(response_text: str, tokens_used: int):
    """Displays the bot response in a nicely formatted panel"""
    console.print(f"\n[bold green]CalderBot:[/bold green]")
    # Markdown rendering — if the AI responds with markdown
    # (like code blocks), Rich will render it beautifully
    console.print(Markdown(response_text))
    console.print(
        f"[dim]Tokens this message: {tokens_used} · "
        f"Total session tokens: {total_tokens}[/dim]"
    )

def show_history():
    """Shows the full conversation history"""
    if not history:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return
    console.print(Rule("Conversation History", style="yellow"))
    for i, msg in enumerate(history):
        if isinstance(msg, HumanMessage):
            console.print(f"[bold cyan]You:[/bold cyan] {msg.content}")
        else:
            console.print(f"[bold green]Bot:[/bold green] {msg.content[:100]}...")
    console.print(Rule(style="yellow"))

# ── Main Chat Loop ──────────────────────────────────────────────
def main():
    global total_tokens
    
    display_welcome()
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input == "/exit":
                console.print(f"\n[bold red]Goodbye! Total tokens used: {total_tokens}[/bold red]")
                break
            
            if user_input == "/clear":
                history.clear()
                total_tokens = 0
                console.print("[bold yellow]Conversation cleared! Starting fresh.[/bold yellow]")
                continue
            
            if user_input == "/history":
                show_history()
                continue
            
            if not user_input:
                continue
            
            # Show thinking indicator
            console.print("[dim]Thinking...[/dim]")
            
            # Send to Groq with full history
            response = chain.invoke({
                "history": history,
                "question": user_input
            })
            
            # Extract token usage from response metadata
            tokens_used = response.response_metadata.get(
                "token_usage", {}
            ).get("total_tokens", 0)
            total_tokens += tokens_used
            
            bot_reply = response.content
            
            # Save both messages to history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=bot_reply))
            
            # Display the response
            display_response(bot_reply, tokens_used)
            
            # Show history size
            console.print(f"[dim]Messages in memory: {len(history)}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted. Type /exit to quit properly.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[dim]Please try again.[/dim]")

# ── Entry Point ─────────────────────────────────────────────────
# This means: only run main() if this file is run directly
# not if it's imported by another file
if __name__ == "__main__":
    main()