import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# The same news article for all 5 prompts
NEWS_ARTICLE = """
OpenAI announced GPT-5 today, claiming it achieves human-level performance 
on most professional benchmarks. The model scores 90% on the bar exam, 
88% on medical licensing tests, and 95% on coding challenges. 
CEO Sam Altman stated the model will be available to ChatGPT Plus users 
next week at no extra cost. Critics argue that benchmark performance 
doesn't reflect real-world usefulness. The model uses 10x more compute 
than GPT-4 and required 6 months of training on undisclosed data.
"""

# 5 different system prompts for the same task
PROMPTS = {
    "Prompt 1 - Vague": 
        "Summarize this.",
    
    "Prompt 2 - Role Based": 
        "You are a professional news editor. Summarize this article clearly.",
    
    "Prompt 3 - Structured": 
        """Summarize this news article in exactly 3 bullet points.
        Each bullet must be under 15 words.
        Focus only on facts, no opinions.""",
    
    "Prompt 4 - Audience Focused": 
        """You are explaining this to a 15 year old with no tech background.
        Use simple words. Maximum 3 sentences.
        Avoid technical jargon.""",
    
    "Prompt 5 - Chain of Thought": 
        """First identify the main topic of this article.
        Then identify the 3 most important facts.
        Then identify any controversies or opposing views.
        Finally write a 2 sentence summary covering all of the above."""
}

console.print(Panel("Lab 1.3 — Prompt Engineering A/B Test", style="bold blue"))
console.print("\n[bold]Testing 5 different prompts on the same article...[/bold]\n")

# Store results
results = []

for prompt_name, system_prompt in PROMPTS.items():
    console.print(f"[yellow]Running {prompt_name}...[/yellow]")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Article: {article}\n\nPlease summarize this.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"article": NEWS_ARTICLE})
    
    results.append({
        "name": prompt_name,
        "system_prompt": system_prompt,
        "response": response,
        "word_count": len(response.split())
    })

# Display results
for r in results:
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold green]{r['name']}[/bold green]")
    console.print(f"[dim]System prompt: {r['system_prompt'][:80]}...[/dim]")
    console.print(f"[bold]Word count: {r['word_count']}[/bold]")
    console.print(f"\n{r['response']}")

# Summary table
console.print(f"\n[bold blue]{'='*60}[/bold blue]")
console.print("\n[bold]Summary Comparison:[/bold]\n")

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Prompt", style="cyan")
table.add_column("Word Count", justify="center")
table.add_column("Style")

styles = ["Vague/inconsistent", "Professional tone", 
          "Structured/precise", "Simple/accessible", "Analytical/thorough"]

for r, style in zip(results, styles):
    table.add_row(r['name'], str(r['word_count']), style)

console.print(table)
console.print("\n[bold green]Lab 1.3 Complete![/bold green]")