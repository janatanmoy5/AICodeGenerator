import streamlit as st
import requests
import re
from datetime import datetime

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Academic AI Code Generator")

# -----------------------------
# Read API Key
# -----------------------------
GROQ_API_KEY = ""

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Settings")

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    language = st.selectbox(
        "Language",
        [
            "Python",
            "SQL",
            "R",
            "PHP",
            "HTML",
            "JavaScript",
            "Java",
            "C++"
        ]
    )

    detail = st.selectbox(
        "Detail Level",
        [
            "Detailed",
            "Very Detailed",
            "Production Quality"
        ]
    )

# -----------------------------
# User Input
# -----------------------------
instruction = st.text_area(
    "Enter Instruction",
    height=220,
    placeholder="Write Python code to search a name in a string"
)

existing_code = st.text_area(
    "Optional Existing Code",
    height=150
)

# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt():

    prompt = f"""
You are an expert software engineer.

Programming Language:
{language}

Detail Level:
{detail}

User Request:
{instruction}

Existing Code:
{existing_code}

Requirements:

1. Generate complete runnable code.
2. Include imports.
3. Include comments.
4. Include error handling.
5. Include example usage.
6. Do not omit important logic.
7. Return code only.
"""

    return prompt

# -----------------------------
# Extract Code
# -----------------------------
def extract_code(text):

    matches = re.findall(
        r"```(?:\\w+)?\\n(.*?)```",
        text,
        re.DOTALL
    )

    if matches:
        return "\n\n".join(matches)

    return text

# -----------------------------
# Call Groq
# -----------------------------
def generate_code(prompt):

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
                "content": "Return complete runnable code."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"]

# -----------------------------
# Generate Button
# -----------------------------
if st.button("🚀 Generate Code"):

    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY not found in Streamlit Secrets."
        )
        st.stop()

    if not instruction.strip():
        st.warning("Please enter an instruction.")
        st.stop()

    try:

        with st.spinner("Generating code..."):

            prompt = build_prompt()

            raw_output = generate_code(prompt)

            final_code = extract_code(raw_output)

        st.success("Code Generated")

        st.subheader("Generated Code")

        st.code(final_code)

        filename = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".txt"
        )

        st.download_button(
            label="Download Code",
            data=final_code,
            file_name=filename,
            mime="text/plain"
        )

    except Exception as e:

        st.error(str(e))

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    """
### Streamlit Cloud Setup

Add this secret:

```toml
GROQ_API_KEY = "your_api_key"
