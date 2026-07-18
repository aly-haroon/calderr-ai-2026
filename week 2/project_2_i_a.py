import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

# ── STAGE 1: PYDANTIC MODEL ────────────────────────────────────
# This defines the exact shape of our internship application.
# Every field has a description — the AI reads these to know
# exactly what value to put in each field.

class InternshipApplication(BaseModel):
    """Complete internship application form"""

    # PERSONAL INFO
    full_name: str = Field(
        description="Full name of the applicant"
    )
    email: str = Field(
        description="Valid email address of the applicant"
    )
    phone: str = Field(
        description="Phone number with country code if provided"
    )
    city: str = Field(
        description="City where the applicant is located"
    )

    # ACADEMIC INFO
    university: str = Field(
        description="Full name of the university"
    )
    degree_program: str = Field(
        description="Degree program e.g. BS Computer Science, "
                   "BS Software Engineering"
    )
    graduation_year: int = Field(
        description="Expected graduation year as integer e.g. 2027",
        ge=2024,
        le=2030
    )
    cgpa: float = Field(
        description="CGPA on a 4.0 scale. "
                   "Convert if given on different scale.",
        ge=0.0,
        le=4.0
    )

    # EXPERIENCE & SKILLS
    skills: List[str] = Field(
        description="List of technical skills. "
                   "Each skill as a separate string."
    )
    previous_internships: bool = Field(
        description="True if applicant has done internships before, "
                   "False if this is their first"
    )

    # INTERNSHIP PREFERENCES
    domain: str = Field(
        description="Preferred domain or role e.g. "
                   "AI/ML, Web Development, Data Science, "
                   "DevOps, Mobile Development"
    )
    work_type: str = Field(
        description="Preferred work arrangement: "
                   "'remote', 'onsite', or 'hybrid'"
    )
    start_date: str = Field(
        description="When applicant can start e.g. "
                   "'July 2026', 'Immediately', 'August 1st 2026'"
    )

    # OPTIONAL EXTRAS
    github: Optional[str] = Field(
        default=None,
        description="GitHub profile URL if provided"
    )
    linkedin: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL if provided"
    )

    @field_validator('work_type')
    @classmethod
    def validate_work_type(cls, v):
        """Ensure work_type is always one of our three options"""
        allowed = ['remote', 'onsite', 'hybrid']
        if v.lower() not in allowed:
            return 'onsite'
        return v.lower()

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Basic email validation"""
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()

# ── STAGE 2: MISSING FIELDS DETECTOR ──────────────────────────
# Required fields that MUST be present before filling the form
REQUIRED_FIELDS = [
    "full_name",
    "email", 
    "phone",
    "city",
    "university",
    "degree_program",
    "graduation_year",
    "cgpa",
    "skills",
    "previous_internships",
    "domain",
    "work_type",
    "start_date"
]

# LLM for detecting missing fields
detector_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def detect_missing_fields(user_input: str) -> List[str]:
    """
    Analyzes user input and returns list of missing required fields.
    
    Uses LLM to understand natural language — so it can detect
    that "Ali Haroon" satisfies full_name even without explicit
    labeling.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing an internship application 
input to find missing information.

Required fields:
- full_name: applicant's full name
- email: email address
- phone: phone number
- city: current city
- university: university name
- degree_program: degree being pursued
- graduation_year: expected graduation year
- cgpa: grade point average
- skills: technical skills list
- previous_internships: whether they had internships before
- domain: preferred work domain (AI, Web Dev etc)
- work_type: remote/onsite/hybrid preference
- start_date: when they can start

Analyze the input carefully.
Return ONLY a comma separated list of missing field names.
If nothing is missing return: NONE
Do not explain, just list missing fields or NONE."""),
        ("user", """Analyze this application input carefully.

Look for these specific things:
- "no previous internships" or "first internship" = previous_internships IS provided (False)
- "github.com/..." = github IS provided
- Any CGPA or GPA number = cgpa IS provided
- Any year like 2027 = graduation_year IS provided
- Skills listed anywhere = skills IS provided
- Remote/onsite/hybrid mentioned anywhere = work_type IS provided
- Any domain like AI/ML, Web Dev = domain IS provided
- Any start date or availability = start_date IS provided

Input to analyze:
{user_input}""")
    ])
    
    chain = prompt | detector_llm | StrOutputParser()
    result = chain.invoke({"user_input": user_input})
    
    # Parse the response
    result = result.strip()
    if result == "NONE" or result == "":
        return []
    
    # Convert comma separated string to list
    missing = [
        field.strip() 
        for field in result.split(",")
        if field.strip() in REQUIRED_FIELDS
    ]
    return missing

# ── STAGE 3: CLARIFICATION AGENT ──────────────────────────────

clarification_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,  # slightly creative for natural conversation
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_clarification_questions(
    missing_fields: List[str],
    user_input: str
) -> str:
    """
    Takes list of missing fields and generates natural
    conversational questions to ask the user.
    
    temperature=0.7 here because we want the questions
    to feel natural and friendly, not robotic.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly internship application 
assistant helping someone complete their application.

Generate clear, friendly questions for the missing information.
Number each question.
Keep it conversational and encouraging.
Don't mention field names directly — ask naturally.

Examples:
- Instead of "Provide email" → "What's your email address?"
- Instead of "Provide cgpa" → "What's your current CGPA?"
- Instead of "Provide previous_internships" → 
  "Have you done any internships before?"
- Instead of "Provide work_type" →
  "Do you prefer working remotely, onsite, or hybrid?"
- Instead of "Provide start_date" →
  "When would you be available to start?"
"""),
        ("user", """The applicant provided this so far:
{user_input}

Still need these details: {missing_fields}

Generate friendly questions to collect the missing info:""")
    ])
    
    chain = prompt | clarification_llm | StrOutputParser()
    
    return chain.invoke({
        "user_input": user_input,
        "missing_fields": ", ".join(missing_fields)
    })

def collect_clarifications(
    missing_fields: List[str],
    original_input: str
) -> str:
    """
    Interactive loop that:
    1. Shows questions to user
    2. Collects their answers
    3. Returns combined input (original + answers)
    """
    console.print(
        "\n[bold yellow]I need a few more details "
        "to complete your application:[/bold yellow]\n"
    )
    
    # Generate natural questions
    questions = generate_clarification_questions(
        missing_fields,
        original_input
    )
    console.print(f"[cyan]{questions}[/cyan]\n")
    
    # Collect user's answers
    console.print(
        "[dim]Please answer all questions above "
        "(you can answer in one message):[/dim]"
    )
    answers = input("\nYour answers: ").strip()
    
    # Combine original input with new answers
    combined = f"""
Original information provided:
{original_input}

Additional information provided:
{answers}
"""
    return combined

# ── TEST STAGE 3 ───────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Panel(
        "[bold blue]Stage 3: Clarification Agent[/bold blue]",
        style="blue"
    ))
    
    # Simulate incomplete input
    incomplete = "I am Ali Haroon, CS student at FAST Islamabad, I know Python and ML"
    
    console.print(f"[dim]Input: {incomplete}[/dim]\n")
    
    # Detect missing fields
    missing = detect_missing_fields(incomplete)
    console.print(
        f"[red]Missing {len(missing)} fields: "
        f"{missing}[/red]\n"
    )
    
    # Generate and show clarification questions
    questions = generate_clarification_questions(
        missing, incomplete
    )
    console.print(
        "[bold yellow]Clarification questions "
        "generated:[/bold yellow]"
    )
    console.print(f"[cyan]{questions}[/cyan]")