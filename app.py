import streamlit as st
import requests
import re
from datetime import datetime

st.set_page_config(
    page_title="Academic AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Academic AI Code Generator")
st.write("Generate ready-to-run code from any instruction using an AI model.")

# -----------------------------
# API KEY from Streamlit Secrets
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("AI Settings")

model = st.sidebar.selectbox(
    "Choose Model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "deepseek-r1-distill-llama-70b"
    ]
)

language = st.sidebar.selectbox(
    "Programming Language",
    [
        "Python",
        "Streamlit Python",
        "Flask Python",
        "R",
        "SQL",
        "PHP",
        "HTML",
        "JavaScript",
        "Java",
        "C++",
        "Bash",
        "Other"
    ]
)

code_type = st.sidebar.selectbox(
    "Code Type",
    [
        "Complete application",
        "Single script",
        "Function only",
        "Debug existing code",
        "Modify existing code",
        "Machine learning pipeline",
        "Bioinformatics pipeline",
        "Cheminformatics pipeline",
        "SQL query",
        "Web application"
    ]
)

detail_level = st.sidebar.selectbox(
    "Detail Level",
    [
        "Detailed",
        "Very detailed",
        "Production quality"
    ]
)

temperature = st.sidebar.slider(
    "Creativity",
    0.0,
    1.0,
    0.2,
    0.1
)

max_tokens = st.sidebar.slider(
    "Maximum Output Tokens",
    1000,
    8000,
    4000,
    1000
)

# -----------------------------
# Main input
# -----------------------------
instruction = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="""
Example:
Write Python code to input a string and search the name Tanmoy.

Example:
Create a Streamlit app to upload CSV and plot PCA.

Example:
Write SQL query to retrieve HLA typing using sample number.
"""
)

existing_code = st.text_area(
    "Optional: Paste existing code for debugging or modification",
    height=160
)

# -----------------------------
# Helper functions
# -----------------------------
def build_prompt():
    return f"""
You are an expert academic software engineer.

Programming language:
{language}

Code type:
{code_type}

Detail level:
{detail_level}

User instruction:
{instruction}

Existing code:
{existing_code}

Requirements:
1. Understand the request.
2. Create the full logic.
3. Generate complete ready-to-run code.
4. Include imports.
5. Include comments.
6. Include error handling.
7. Avoid placeholder code.
8. If Streamlit app is requested, write complete app.py.
9. If ML is requested, include preprocessing, training, evaluation, and saving.
10. Return code only.
"""

def extract_code(text):
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks).strip()
    return text.strip()

def get_extension(lang):
    mapping = {
        "Python": "py",
        "Streamlit Python": "py",
        "Flask Python": "py",
        "R": "R",
        "SQL": "sql",
        "PHP": "php",
        "HTML": "html",
        "JavaScript": "js",
        "Java": "java",
        "C++": "cpp",
        "Bash": "sh",
        "Other": "txt"
    }
    return mapping.get(lang, "txt")

def generate_code(prompt):
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert coding assistant. Return complete runnable code only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]

# -----------------------------
# Generate button
# -----------------------------
if st.button("🚀 Generate Code", use_container_width=True):

    if not instruction.strip():
        st.warning("Please enter a coding instruction.")
        st.stop()

    try:
        with st.spinner("Generating code..."):
            prompt = build_prompt()
            raw_output = generate_code(prompt)
            clean_code = extract_code(raw_output)

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(clean_code)

        filename = f"generated_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{get_extension(language)}"

        st.download_button(
            "⬇️ Download Code",
            clean_code,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("View full AI response"):
            st.write(raw_output)

    except Exception as e:
        st.error(str(e))

# -----------------------------
# Instructions
# -----------------------------
st.divider()

st.subheader("Streamlit Cloud Setup")

st.markdown(
    """
In your Streamlit Cloud app, go to:

**Manage App → Settings → Secrets**

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
