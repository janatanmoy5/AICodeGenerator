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
st.write("Generate ready-to-run code from any instruction using Groq API.")

# -----------------------------
# Read Groq API key
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
    "Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
)

language = st.sidebar.selectbox(
    "Programming Language",
    [
        "Python",
        "Streamlit Python",
        "Flask Python",
        "SQL",
        "R",
        "PHP",
        "HTML",
        "JavaScript",
        "Java",
        "C++",
        "Bash",
        "Other"
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
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Maximum Output Tokens",
    min_value=1000,
    max_value=8000,
    value=4000,
    step=1000
)

# -----------------------------
# Main input
# -----------------------------
instruction = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: Write Python code to input a string and search the name Tanmoy."
)

existing_code = st.text_area(
    "Optional: Paste existing code for debugging or modification",
    height=150
)

# -----------------------------
# Functions
# -----------------------------
def build_prompt():
    parts = []

    parts.append("You are an expert academic software engineer.")
    parts.append("Generate complete, ready-to-run code.")
    parts.append("")
    parts.append("Programming language: " + language)
    parts.append("Detail level: " + detail_level)
    parts.append("")
    parts.append("User instruction:")
    parts.append(instruction)
    parts.append("")
    parts.append("Existing code if provided:")
    parts.append(existing_code)
    parts.append("")
    parts.append("Requirements:")
    parts.append("1. Understand the request.")
    parts.append("2. Create the full logic.")
    parts.append("3. Generate complete runnable code.")
    parts.append("4. Include all imports.")
    parts.append("5. Include comments.")
    parts.append("6. Include error handling.")
    parts.append("7. Avoid placeholder code.")
    parts.append("8. If Streamlit app is requested, write complete app.py.")
    parts.append("9. If machine learning is requested, include preprocessing, training, evaluation, and saving.")
    parts.append("10. Return code only.")

    return "\n".join(parts)


def extract_code(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return "\n\n".join(matches).strip()

    return text.strip()


def get_file_extension(lang):
    mapping = {
        "Python": "py",
        "Streamlit Python": "py",
        "Flask Python": "py",
        "SQL": "sql",
        "R": "R",
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
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": "Bearer " + GROQ_API_KEY,
        "Content-Type": "application/json"
    }

    system_message = "You are an expert coding assistant. Return complete runnable code only."

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message
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
# Generate Button
# -----------------------------
if st.button("🚀 Generate Code", use_container_width=True):

    if not instruction.strip():
        st.warning("Please enter an instruction.")
        st.stop()

    try:
        prompt = build_prompt()

        with st.spinner("Generating code..."):
            raw_output = generate_code(prompt)

        clean_code = extract_code(raw_output)

        st.success("Code generated successfully.")

        st.subheader("Generated Code")

        st.code(clean_code)

        extension = get_file_extension(language)

        file_name = "generated_code_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "." + extension

        st.download_button(
            label="⬇️ Download Code",
            data=clean_code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("View full AI response"):
            st.write(raw_output)

    except Exception as e:
        st.error(str(e))

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.subheader("Streamlit Cloud Setup")

st.write("Add this secret in Streamlit Cloud:")
st.code('GROQ_API_KEY = "your_groq_api_key_here"', language="toml")

st.write("requirements.txt:")
st.code("streamlit\nrequests", language="text")
