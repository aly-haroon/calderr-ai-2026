import os
import time
import random
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

# ── LOGGING SETUP ──────────────────────────────────────────────
# Logging records everything that happens — errors, retries, 
# successes. Essential for debugging production systems.
logging.basicConfig(
    filename='agent_log.txt',  # saves to file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── RETRY DECORATOR ────────────────────────────────────────────
# This is the most important concept in this lab.
# A decorator that adds retry + exponential backoff
# to ANY function automatically.

def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """
    Decorator factory that adds retry logic to any function.
    
    max_attempts: how many times to try before giving up
    base_delay: starting wait time in seconds
    
    Exponential backoff formula:
    wait = base_delay * (2 ^ attempt_number)
    Attempt 1 → wait 1s
    Attempt 2 → wait 2s  
    Attempt 3 → wait 4s
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # Try to run the function
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        # Log successful recovery
                        logger.info(
                            f"{func.__name__} succeeded "
                            f"on attempt {attempt}"
                        )
                        console.print(
                            f"[green]✅ Recovered on attempt "
                            f"{attempt}![/green]"
                        )
                    return result
                    
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt} "
                        f"failed: {str(e)}"
                    )
                    console.print(
                        f"[yellow]⚠️ Attempt {attempt} failed: "
                        f"{str(e)}[/yellow]"
                    )
                    
                    if attempt < max_attempts:
                        # Exponential backoff
                        wait_time = base_delay * (2 ** (attempt - 1))
                        console.print(
                            f"[dim]Waiting {wait_time}s "
                            f"before retry...[/dim]"
                        )
                        time.sleep(wait_time)
            
            # All attempts failed
            logger.error(
                f"{func.__name__} failed after "
                f"{max_attempts} attempts: {str(last_error)}"
            )
            raise Exception(
                f"Failed after {max_attempts} attempts. "
                f"Last error: {str(last_error)}"
            )
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# ── UNRELIABLE TOOLS ───────────────────────────────────────────
# These tools simulate real world failures.
# They randomly fail to test our retry logic.

@tool
def unreliable_weather_api(city: str) -> str:
    """Get weather for a city. Sometimes fails due to API issues."""
    
    # Simulate 60% failure rate to test retry logic
    if random.random() < 0.6:
        raise Exception(f"Weather API timeout for {city}")
    
    # Fake weather data
    weather_data = {
        "karachi": "Sunny, 35°C, Humidity: 70%",
        "lahore": "Partly cloudy, 38°C, Humidity: 45%",
        "islamabad": "Clear, 32°C, Humidity: 55%",
        "london": "Rainy, 15°C, Humidity: 85%",
    }
    city_lower = city.lower()
    weather = weather_data.get(
        city_lower, 
        f"Partly cloudy, 25°C"
    )
    return f"Weather in {city}: {weather}"

@tool  
def unreliable_stock_api(symbol: str) -> str:
    """Get stock price for a symbol. Occasionally unavailable."""
    
    # Simulate 50% failure rate
    if random.random() < 0.5:
        raise Exception(
            f"Stock API rate limit exceeded for {symbol}"
        )
    
    # Fake stock data
    stocks = {
        "AAPL": "$189.50 (+1.2%)",
        "GOOGL": "$141.20 (-0.5%)",
        "MSFT": "$378.90 (+0.8%)",
        "TSLA": "$245.30 (-2.1%)",
    }
    price = stocks.get(
        symbol.upper(), 
        f"${random.uniform(50, 500):.2f}"
    )
    return f"{symbol.upper()} Stock Price: {price}"

@tool
def backup_weather_tool(city: str) -> str:
    """Backup weather tool using cached data. Always works."""
    return (
        f"Weather in {city} (cached data): "
        f"Approximately 25-35°C, typical conditions"
    )

@tool
def backup_stock_tool(symbol: str) -> str:
    """Backup stock tool using cached data. Always works."""
    return (
        f"{symbol.upper()} (cached price): "
        f"Market data temporarily unavailable. "
        f"Last known price was in normal trading range."
    )

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# ── RELIABLE WRAPPERS ──────────────────────────────────────────
# Wrap unreliable tools with retry + fallback logic

def get_weather_with_recovery(city: str) -> str:
    """
    Try primary weather API with retries.
    If all retries fail, use backup tool.
    """
    console.print(
        f"[cyan]Getting weather for {city}...[/cyan]"
    )
    logger.info(f"Weather request for: {city}")
    
    # Apply retry decorator dynamically
    @with_retry(max_attempts=3, base_delay=0.5)
    def try_primary():
        return unreliable_weather_api.invoke({"city": city})
    
    try:
        result = try_primary()
        logger.info(f"Primary weather API succeeded for {city}")
        return result
    except Exception as e:
        # Primary failed after all retries — use backup
        console.print(
            f"[red]Primary weather API failed. "
            f"Using backup...[/red]"
        )
        logger.warning(
            f"Falling back to backup weather for {city}"
        )
        return backup_weather_tool.invoke({"city": city})

def get_stock_with_recovery(symbol: str) -> str:
    """
    Try primary stock API with retries.
    If all retries fail, use backup tool.
    """
    console.print(
        f"[cyan]Getting stock price for {symbol}...[/cyan]"
    )
    logger.info(f"Stock request for: {symbol}")
    
    @with_retry(max_attempts=3, base_delay=0.5)
    def try_primary():
        return unreliable_stock_api.invoke({"symbol": symbol})
    
    try:
        result = try_primary()
        logger.info(
            f"Primary stock API succeeded for {symbol}"
        )
        return result
    except Exception as e:
        console.print(
            f"[red]Primary stock API failed. "
            f"Using backup...[/red]"
        )
        logger.warning(
            f"Falling back to backup stock for {symbol}"
        )
        return backup_stock_tool.invoke({"symbol": symbol})

# ── TOOLS FOR AGENT ────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get current weather for any city.
    Use this when asked about weather conditions."""
    return get_weather_with_recovery(city)

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol like AAPL, GOOGL.
    Use this when asked about stock prices."""
    return get_stock_with_recovery(symbol)

# ── LLM + AGENT SETUP ──────────────────────────────────────────

tools = [get_weather, get_stock_price, calculate]
tool_map = {t.name: t for t in tools}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)

def run_agent(query: str) -> str:
    """Agent loop with full error recovery"""
    messages = [HumanMessage(content=query)]
    logger.info(f"New query: {query}")
    
    for iteration in range(5):
        try:
            response = llm.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                logger.info(f"Final answer: {response.content}")
                return response.content
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                console.print(
                    f"[yellow]→ Calling: {tool_name}[/yellow]"
                )
                
                try:
                    if tool_name in tool_map:
                        result = tool_map[tool_name].invoke(
                            tool_args
                        )
                    else:
                        result = f"Tool '{tool_name}' not found"
                except Exception as e:
                    # Tool failed even after retries
                    result = (
                        f"Tool failed: {str(e)}. "
                        f"Please provide best answer without it."
                    )
                    logger.error(
                        f"Tool {tool_name} failed: {str(e)}"
                    )
                
                console.print(
                    f"[green]  Result: {str(result)[:100]}"
                    f"[/green]"
                )
                
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                ))
                
        except Exception as e:
            logger.error(f"Agent loop error: {str(e)}")
            return f"Agent error: {str(e)}"
    
    return "Could not complete request"

# ── TEST QUERIES ────────────────────────────────────────────────

TEST_QUERIES = [
    "What is the weather in Karachi?",
    "What is the current price of AAPL stock?",
    "What is the weather in London and the GOOGL stock price?",
    "What is 15% of 50000?",
]

# ── MAIN ────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold blue]Lab 2.3 — Error Recovery Agent[/bold blue]\n"
        "Retry logic + exponential backoff + fallback tools",
        style="blue"
    ))

    for i, query in enumerate(TEST_QUERIES, 1):
        console.print(
            f"\n[bold blue]{'='*60}[/bold blue]"
        )
        console.print(
            f"[bold cyan]Query {i}:[/bold cyan] {query}"
        )
        console.print(
            f"[bold blue]{'='*60}[/bold blue]"
        )
        
        answer = run_agent(query)
        console.print(
            f"\n[bold green]Answer:[/bold green] {answer}"
        )

    # Show log summary
    console.print(
        f"\n[bold yellow]All attempts logged to: "
        f"agent_log.txt[/bold yellow]"
    )
    console.print(
        "[dim]Open agent_log.txt to see full retry history[/dim]"
    )

if __name__ == "__main__":
    main()