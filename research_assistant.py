import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()
print(os.getenv("GROQ_API_KEY"))
# ── PYDANTIC MODELS ────────────────────────────────────────────

class ResearchSubtopic(BaseModel):
    title: str = Field(description="Title of the subtopic")
    research_question: str = Field(description="Specific question to research")

class ResearchPlan(BaseModel):
    main_topic: str = Field(description="The main research topic")
    subtopics: List[ResearchSubtopic] = Field(description="List of subtopics to research")
    research_goal: str = Field(description="Overall goal of this research")

class ResearchFinding(BaseModel):
    subtopic: str = Field(description="The subtopic that was researched")
    content: str = Field(description="Detailed research findings")
    confidence_score: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    key_points: List[str] = Field(description="3-5 key points from this research")

class FinalReport(BaseModel):
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary of all findings")
    findings: List[ResearchFinding] = Field(description="All research findings")
    conclusion: str = Field(description="Final conclusion and insights")
    overall_confidence: float = Field(description="Overall confidence score", ge=0, le=1)

# ── LLM SETUP ─────────────────────────────────────────────────

def get_llm(temperature: float = 0):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ── STAGE 2: PLANNER AGENT ─────────────────────────────────────

def create_research_plan(question: str, num_subtopics: int = 4) -> ResearchPlan:
    structured_llm = get_llm(0).with_structured_output(ResearchPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert research planner.
Break the research question into exactly {num_subtopics} subtopics.
Rules:
- Each subtopic must be distinct
- Flow from fundamentals to advanced concepts
- Each subtopic needs a specific focused research question"""),
        ("user", "Research question: {question}")
    ])
    chain = prompt | structured_llm
    return chain.invoke({"question": question})

# ── STAGE 3: RESEARCH LOOP ─────────────────────────────────────

def research_subtopic(subtopic: ResearchSubtopic, main_topic: str) -> ResearchFinding:
    structured_llm = get_llm(0.3).with_structured_output(ResearchFinding)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert researcher in AI and technology.
Research the subtopic thoroughly and return structured findings.
Rules:
- content must be detailed (minimum 150 words)
- key_points must be exactly 4 bullet points under 20 words each
- confidence_score: 0.9-1.0 established facts, 0.7-0.9 evolving,
  0.5-0.7 debated, 0.3-0.5 speculative"""),
        ("user", """Main topic: {main_topic}
Subtopic: {subtopic_title}
Question: {research_question}""")
    ])
    chain = prompt | structured_llm
    return chain.invoke({
        "main_topic": main_topic,
        "subtopic_title": subtopic.title,
        "research_question": subtopic.research_question
    })

def run_research_loop(plan: ResearchPlan, progress_callback=None) -> List[ResearchFinding]:
    findings = []
    for i, subtopic in enumerate(plan.subtopics, 1):
        if progress_callback:
            progress_callback(i, len(plan.subtopics), subtopic.title)
        finding = research_subtopic(subtopic, plan.main_topic)
        findings.append(finding)
    return findings

# ── STAGE 4: SYNTHESIS AGENT ───────────────────────────────────

def synthesize_report(plan: ResearchPlan, findings: List[ResearchFinding]) -> FinalReport:
    structured_llm = get_llm(0.5).with_structured_output(FinalReport)
    findings_text = ""
    for f in findings:
        findings_text += f"\n\nSubtopic: {f.subtopic}\n"
        findings_text += f"Confidence: {f.confidence_score}\n"
        findings_text += f"Content: {f.content}\n"
        findings_text += f"Key Points: {', '.join(f.key_points)}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research synthesizer.
Take all research findings and produce a comprehensive final report.
Rules:
- summary must be 3-4 sentences covering the entire research
- conclusion must provide actionable insights
- overall_confidence is the average of all finding confidence scores
- title must be professional and descriptive"""),
        ("user", """Main topic: {main_topic}
Research goal: {research_goal}

All findings:
{findings_text}

Please synthesize into a final report.""")
    ])
    chain = prompt | structured_llm
    return chain.invoke({
        "main_topic": plan.main_topic,
        "research_goal": plan.research_goal,
        "findings_text": findings_text
    })

# ── STAGE 5: STREAMLIT UI ──────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Agentic Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Agentic Research Assistant")
    st.caption("Powered by Groq + LangChain | CalderR Internship 2026")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        num_subtopics = st.slider("Number of subtopics", 2, 5, 4)
        st.info("Each subtopic = 1 Groq API call")
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("1. Planner breaks your question into subtopics")
        st.markdown("2. Research loop investigates each subtopic")
        st.markdown("3. Synthesis agent combines all findings")
        st.markdown("4. Final report is displayed here")

    # Input
    question = st.text_input(
        "Enter your research question:",
        placeholder="e.g. What is Agentic and Generative AI?",
        key="question"
    )

    if st.button("🚀 Start Research", type="primary"):
        if not question:
            st.warning("Please enter a research question!")
            return

        # Stage 2 — Planning
        with st.status("🧠 Planning research strategy...", expanded=True) as status:
            st.write("Analyzing your question...")
            plan = create_research_plan(question, num_subtopics)
            st.write(f"✅ Research plan created!")
            st.write(f"**Topic:** {plan.main_topic}")
            st.write(f"**Goal:** {plan.research_goal}")
            st.write("**Subtopics identified:**")
            for i, s in enumerate(plan.subtopics, 1):
                st.write(f"{i}. {s.title}")
            status.update(label="✅ Research plan ready!")

        # Stage 3 — Research Loop
        findings = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, subtopic in enumerate(plan.subtopics, 1):
            status_text.text(f"🔍 Researching {i}/{len(plan.subtopics)}: {subtopic.title}...")
            progress_bar.progress(i / len(plan.subtopics) * 0.7)
            finding = research_subtopic(subtopic, plan.main_topic)
            findings.append(finding)

        # Stage 4 — Synthesis
        status_text.text("🔗 Synthesizing all findings...")
        progress_bar.progress(0.9)
        report = synthesize_report(plan, findings)
        progress_bar.progress(1.0)
        status_text.text("✅ Research complete!")

        # Display Report
        st.markdown("---")
        st.header(f"📄 {report.title}")

        # Summary
        st.subheader("📋 Executive Summary")
        st.info(report.summary)

        # Overall confidence
        confidence_color = "green" if report.overall_confidence >= 0.7 else "orange"
        st.metric(
            "Overall Confidence Score",
            f"{report.overall_confidence:.2f}",
            help="How confident the AI is about the research findings"
        )

        # Individual findings
        st.subheader("🔬 Research Findings")
        for i, finding in enumerate(report.findings, 1):
            with st.expander(
                f"{i}. {finding.subtopic} — Confidence: {finding.confidence_score:.2f}",
                expanded=True
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Detailed Findings:**")
                    st.write(finding.content)
                with col2:
                    st.markdown("**Key Points:**")
                    for point in finding.key_points:
                        st.markdown(f"• {point}")
                    st.metric("Confidence", f"{finding.confidence_score:.2f}")

        # Conclusion
        st.subheader("🎯 Conclusion")
        st.success(report.conclusion)

        # Architecture diagram
        st.markdown("---")
        st.subheader("🏗️ Agent Architecture Used")
        st.code("""
Question Input
    → Planner Agent (1 Groq call)
        → Sequential Research Loop (4 Groq calls)
            → Synthesis Agent (1 Groq call)  
                → Final Report (Streamlit UI)
        """)

if __name__ == "__main__":
    main()