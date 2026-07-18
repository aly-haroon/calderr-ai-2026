"""
Automated Data Analysis Agent - core modules package.

This package contains the pipeline stages:
    schema_analyzer  -> inspects an uploaded DataFrame
    question_parser  -> normalizes/validates the user's natural language question
    code_generator    -> asks Groq (Llama 3.3 70B) to write pandas code
    code_executor     -> runs that code inside a restricted sandbox
    visualization     -> auto-detects and builds an appropriate chart
    report_builder    -> assembles everything into a shareable report
"""
