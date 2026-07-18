import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

# ── TOOLS ──────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression like '2 + 2' or '150 * 0.15'"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_date() -> str:
    """Get today's current date and time"""
    now = datetime.now()
    return f"Today is {now.strftime('%B %d, %Y at %H:%M:%S')}"

@tool
def search_database(query: str) -> str:
    """Search company database for products, pricing, refund policy, employees, internship info"""
    DATABASE = {
        "products": """
            CalderR Products:
            - AgentOS: Agentic AI platform ($999/month)
            - FlowBuilder: Visual agent workflow builder ($299/month)
            - EvalKit: RAG evaluation framework (open source)
        """,
        "pricing": """
            CalderR Pricing:
            - Starter: Free (100 API calls/day)
            - Pro: $99/month (10,000 calls/day)
            - Enterprise: Custom pricing
        """,
        "refund": """
            Refund Policy:
            - 30 day money back guarantee
            - No questions asked refunds within 30 days
            - Contact support@calderr.ai for refunds
        """,
        "employees": """
            CalderR Team:
            - CEO: Sarah Chen
            - CTO: Ahmed Khan
            - Head of AI: Dr. Priya Patel
        """,
        "internship": """
            CalderR Internship 2026:
            - 10 week program
            - 20 hours per week
            - Focus on agentic AI engineering
        """
    }
    query_lower = query.lower()
    results = []
    for key, value in DATABASE.items():
        if key in query_lower or any(
            word in value.lower() 
            for word in query_lower.split()
        ):
            results.append(value.strip())
    return "\n".join(results) if results else "No info found."

@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into 50 words or less"""
    words = text.split()
    if len(words) <= 50:
        return text
    # Simple extractive summary — first 2 sentences
    sentences = text.split('.')
    return '. '.join(sentences[:2]) + '.'

@tool
def classify_sentiment(text: str) -> str:
    """Classify sentiment of text as POSITIVE, NEGATIVE, or NEUTRAL"""
    positive_words = [
        "great", "excellent", "amazing", "good", "happy",
        "love", "best", "fantastic", "wonderful", "perfect",
        "awesome", "brilliant", "outstanding"
    ]
    negative_words = [
        "bad", "terrible", "awful", "poor", "hate",
        "worst", "horrible", "disappointing", "failed",
        "broken", "useless", "wrong"
    ]
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg:
        return f"POSITIVE (confidence: {min(60 + pos*10, 99)}%)"
    elif neg > pos:
        return f"NEGATIVE (confidence: {min(60 + neg*10, 99)}%)"
    return "NEUTRAL (confidence: 60%)"

# ── TOOL REGISTRY ───────────────────────────────────────────────
# Map tool names to actual functions for easy lookup
tools = [calculate, get_current_date, 
         search_database, summarize_text, classify_sentiment]
tool_map = {t.name: t for t in tools}

# ── LLM SETUP ──────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)

# ── AGENT LOOP ──────────────────────────────────────────────────
def run_agent(query: str) -> str:
    """
    Manual tool calling loop:
    1. Send query to LLM
    2. If LLM wants to call tools, run them
    3. Send results back to LLM
    4. Repeat until LLM gives final answer
    """
    messages = [HumanMessage(content=query)]
    
    for iteration in range(5):  # max 5 iterations
        response = llm.invoke(messages)
        messages.append(response)
        
        # If no tool calls, LLM has final answer
        if not response.tool_calls:
            return response.content
        
        # Run each tool the LLM requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]
            
            console.print(f"[yellow]→ Calling tool: {tool_name}[/yellow]")
            console.print(f"[dim]  Args: {tool_args}[/dim]")
            
            # Look up and run the tool
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Tool '{tool_name}' not found"
            
            console.print(f"[green]  Result: {result}[/green]")
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            ))
    
    return "Max iterations reached"

# ── TEST QUERIES ────────────────────────────────────────────────
TEST_QUERIES = [
    "What is 15% of 847?",
    "What is today's date?",
    "What products does CalderR offer and what do they cost?",
    "What is the CalderR refund policy?",
    "What is 234 * 567 and what is today's date?",
    """Analyze the sentiment of this: 'The product is absolutely 
    amazing! Best purchase I ever made. Fantastic quality!'""",
    """Summarize this: 'Artificial intelligence is transforming 
    industries across the world. From healthcare to finance, 
    AI systems are being deployed to automate tasks and improve 
    decision making across every sector.'"""
]

# ── MAIN ────────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold blue]Lab 2.2 — Multi Tool Research Agent[/bold blue]\n"
        "5 tools: calculate, date, search, summarize, sentiment",
        style="blue"
    ))

    for i, query in enumerate(TEST_QUERIES, 1):
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold cyan]Query {i}:[/bold cyan] {query[:80]}")
        console.print(f"[bold blue]{'='*60}[/bold blue]")

        try:
            answer = run_agent(query)
            console.print(f"\n[bold green]Answer:[/bold green] {answer}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()