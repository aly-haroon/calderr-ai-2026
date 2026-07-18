import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

# ── PYDANTIC MODELS ────────────────────────────────────────────
# Notice how detailed the Field descriptions are.
# The AI reads these to understand what to extract.

class SalaryRange(BaseModel):
    """Represents salary information"""
    minimum: Optional[int] = Field(
        default=None,
        description="Minimum salary as a plain integer. "
                   "Convert any shorthand: 150k = 150000"
    )
    maximum: Optional[int] = Field(
        default=None,
        description="Maximum salary as a plain integer. "
                   "Convert any shorthand: 200k = 200000"
    )
    currency: str = Field(
        default="USD",
        description="Currency code. Detect from context: "
                   "PKR for Pakistan, USD for US, GBP for UK etc."
    )
    period: str = Field(
        default="yearly",
        description="Payment period: 'yearly', 'monthly', or 'hourly'"
    )

class JobPosting(BaseModel):
    """Complete structured job posting"""
    title: str = Field(
        description="Job title. Infer if not explicitly stated."
    )
    company: str = Field(
        default="Not mentioned",
        description="Company name. Use 'Not mentioned' if not found."
    )
    location: str = Field(
        default="Not mentioned",
        description="City or country of the job."
    )
    salary: SalaryRange = Field(
        description="Salary information extracted from posting."
    )
    skills: List[str] = Field(
        description="List of required or preferred technical skills. "
                   "Each skill as a separate string."
    )
    experience_years: Optional[int] = Field(
        default=None,
        description="Minimum years of experience required as integer."
    )
    remote: bool = Field(
        description="True if any remote work is mentioned or implied."
    )
    remote_type: str = Field(
        default="onsite",
        description="One of: 'fully_remote', 'hybrid', 'onsite'"
    )
    seniority: str = Field(
        default="mid",
        description="One of: 'junior', 'mid', 'senior', 'lead'. "
                   "Infer from context and experience required."
    )
    summary: str = Field(
        description="One sentence summary of the role in plain English."
    )

    @field_validator('remote_type')
    @classmethod
    def validate_remote_type(cls, v):
        """Ensures remote_type is always one of our expected values"""
        allowed = ['fully_remote', 'hybrid', 'onsite']
        if v not in allowed:
            return 'onsite'  # default if AI returns something unexpected
        return v

    @field_validator('seniority')\
    @classmethod
    def validate_seniority(cls, v):
        """Ensures seniority is always one of our expected values"""
        allowed = ['junior', 'mid', 'senior', 'lead']
        if v.lower() not in allowed:
            return 'mid'
        return v.lower()


# ── LLM SETUP ─────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,  # temperature=0 for extraction
                    # we want consistent, deterministic output
                    # not creative interpretation
    api_key=os.getenv("GROQ_API_KEY")
)

# Force LLM to return JobPosting structure
structured_llm = llm.with_structured_output(JobPosting)

# ── EXTRACTION PROMPT ──────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert job posting analyzer.
    
Extract ALL information from the job posting accurately.

Rules:
- Extract exactly what is stated, infer only when obvious
- Convert salary shorthand: 150k = 150000
- Detect currency from context (PKR, USD, GBP etc)
- If information is missing, use the default values
- For skills, extract each technology/skill separately
- Infer seniority from experience required and language used
- Be precise with remote_type: fully_remote/hybrid/onsite"""),
    ("user", "Extract structured information from this job posting:\n\n{job_posting}")
])

chain = prompt | structured_llm

# ── TEST JOB POSTINGS ──────────────────────────────────────────
# These are deliberately messy and varied to test extraction
JOB_POSTINGS = [
    """
    We're looking for a rockstar dev at our Karachi office! 
    Must know Python and ideally React. 
    We pay between 150k-200k PKR monthly. 
    3+ years needed. Remote Fridays allowed.
    Join our fintech startup!
    """,

    """
    Senior Machine Learning Engineer - Remote First
    
    TechCorp Inc. is hiring! We need someone who lives and 
    breathes ML. Requirements: PhD or 7+ years experience,
    expert in PyTorch, TensorFlow, MLOps, Docker, Kubernetes.
    Compensation: $180,000 - $220,000 annually + equity.
    100% remote, US timezone preferred.
    """,

    """
    Junior Frontend Developer wanted for our Lahore office.
    Fresh grads welcome! Training provided.
    Skills: HTML, CSS, JavaScript, React basics
    Salary: Rs 60,000 - 80,000 per month
    Office based, Monday to Friday
    Apply at careers@company.pk
    """,

    """
    LEAD DEVOPS ENGINEER
    London, UK (Hybrid - 2 days office)
    
    We need a DevOps lead to transform our infrastructure.
    10+ years experience required. Must have: AWS, GCP, 
    Terraform, Ansible, Jenkins, GitLab CI. 
    Salary negotiable, range £90k-£120k DOE.
    Immediate start preferred.
    """,

    """
    Data Analyst needed. Excel and SQL skills required.
    Nice to have: Power BI, Python.
    Islamabad based company.
    Salary not disclosed.
    1-2 years experience.
    """
]

# ── DISPLAY FUNCTION ───────────────────────────────────────────
def display_extraction(job_text: str, result: JobPosting, index: int):
    """Shows the extraction results in a clean format"""
    
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold]Job Posting #{index}:[/bold]")
    console.print(f"[dim]{job_text.strip()[:100]}...[/dim]")
    console.print(f"\n[bold green]Extracted Data:[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="white")
    
    table.add_row("Title", result.title)
    table.add_row("Company", result.company)
    table.add_row("Location", result.location)
    table.add_row("Seniority", result.seniority)
    table.add_row("Experience", f"{result.experience_years} years" 
                  if result.experience_years else "Not specified")
    table.add_row("Salary Min", f"{result.salary.currency} "
                  f"{result.salary.minimum:,}" 
                  if result.salary.minimum else "Not disclosed")
    table.add_row("Salary Max", f"{result.salary.currency} "
                  f"{result.salary.maximum:,}"
                  if result.salary.maximum else "Not disclosed")
    table.add_row("Period", result.salary.period)
    table.add_row("Remote", "Yes" if result.remote else "No")
    table.add_row("Remote Type", result.remote_type)
    table.add_row("Skills", ", ".join(result.skills))
    table.add_row("Summary", result.summary)
    
    console.print(table)

# ── MAIN ───────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold blue]Lab 2.1 — Structured Output Extractor[/bold blue]\n"
        "Extracting clean data from messy job postings",
        style="blue"
    ))
    
    console.print(f"\n[bold]Processing {len(JOB_POSTINGS)} job postings...[/bold]\n")
    
    results = []
    
    for i, job_text in enumerate(JOB_POSTINGS, 1):
        console.print(f"[yellow]Extracting job #{i}...[/yellow]")
        
        try:
            result = chain.invoke({"job_posting": job_text})
            results.append(result)
            display_extraction(job_text, result, i)
            
        except Exception as e:
            console.print(f"[red]Error on job #{i}: {e}[/red]")
    
    # Summary statistics
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold green]Summary:[/bold green]")
    console.print(f"Total processed: {len(results)}/{len(JOB_POSTINGS)}")
    
    remote_jobs = sum(1 for r in results if r.remote)
    console.print(f"Remote/Hybrid jobs: {remote_jobs}")
    
    avg_skills = sum(len(r.skills) for r in results) / len(results)
    console.print(f"Average skills per posting: {avg_skills:.1f}")

if __name__ == "__main__":
    main()