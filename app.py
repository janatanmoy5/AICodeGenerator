import os
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="AI Code Generator", layout="wide")

st.title("🤖 AI Code Generator")
st.write("Enter any coding instruction. The AI will create logic and ready-to-run code.")

provider = st.selectbox(
    "Choose AI Provider",
    ["OpenAI", "Ollama Local"]
)

language = st.selectbox(
    "Programming Language",
    ["Python", "PHP", "HTML", "JavaScript", "SQL", "R", "Java", "C++", "Bash", "Other"]
)

if provider == "OpenAI":
    model = st.selectbox(
        "OpenAI Model",
        ["gpt-5.5", "gpt-5", "gpt-4.1"]
    )
else:
    model = st.selectbox(
        "Ollama Model",
        ["qwen2.5-coder", "deepseek-coder", "codellama", "llama3"]
    )

instruction = st.text_area(
    "Write your command / instruction",
    height=220,
    placeholder="""
Example:
Write Python code that takes a string input and searches for a name.

Example:
Create a Streamlit app for chemical similarity search using Tanimoto index.

Example:
Write SQL code to retrieve HLA typing from patient sample number.
"""
)

def build_prompt(user_instruction, language):
    return f"""
You are an expert AI software engineer.

User wants code in: {language}

Instruction:
{user_instruction}

Generate:
1. Complete ready-to-run code
2. Full logic
3. Required imports
4. Error handling
5. Comments
6. Example usage
7. If web app is needed, create complete app.py
8. If multiple files are needed, clearly separate each file

Return code only.
"""

def generate_openai_code(prompt, model):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "ERROR: OPENAI_API_KEY not found. Add it in .env file."

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text

def generate_ollama_code(prompt, model):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=600
    )

    if response.status_code != 200:
        return f"Ollama Error: {response.text}"

    return response.json().get("response", "")

if st.button("Generate Ready-to-Run Code"):
    if not instruction.strip():
        st.warning("Please enter a coding instruction.")
    else:
        prompt = build_prompt(instruction, language)

        with st.spinner("Generating code..."):
            if provider == "OpenAI":
                output = generate_openai_code(prompt, model)
            else:
                output = generate_ollama_code(prompt, model)

        st.subheader("Generated Code")
        st.code(output)

        st.download_button(
            "Download Generated Code",
            output,
            file_name="generated_code.txt",
            mime="text/plain"
        )
