import streamlit as st
import requests
import re
from datetime import datetime

# ==========================================================
# AI Code Generator using Qwen/Ollama
# No OpenAI API required
# ==========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

st.set_page_config(
    page_title="Qwen AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen AI Code Generator")
st.write("Enter any command. Qwen will generate complete code in Python, R, SQL, HTML/CSS, JavaScript, Java, PHP, Bash, and more.")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("AI Model Settings")

model = st.sidebar.selectbox(
    "Choose Qwen Model",
    [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:32b",
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:0.5b"
    ]
)

language = st.sidebar.selectbox(
    "Programming Language",
    [
        "Auto Detect",
        "Python",
        "R",
        "SQL",
        "HTML/CSS",
        "JavaScript",
        "Java",
        "PHP",
        "Bash",
        "C++",
        "Other"
    ]
)

code_type = st.sidebar.selectbox(
    "Code Type",
    [
        "Complete runnable code",
        "Function only",
        "Script",
        "Web application",
        "Streamlit app",
        "Database query",
        "Data analysis",
        "Machine learning",
        "Bioinformatics",
        "Debug or modify code"
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
    "Maximum output tokens",
    1000,
    16000,
    6000,
    1000
)

# ==========================================================
# Input Area
# ==========================================================

command = st.text_area(
    "Enter your command",
    height=220,
    placeholder="""
Examples:

Write Python code to add two numbers.

Write R code for PCA analysis from CSV.

Write SQL query to retrieve HLA typing by sample number.

Create HTML/CSS landing page for a research lab.

Write JavaScript code for a calculator.

Write Java program for student grade calculation.

Create Streamlit app for CSV upload and bar plot.

Create Python machine learning pipeline for classification.
"""
)

existing_code = st.text_area(
    "Optional: Paste existing code for debugging or modification",
    height=160
)

# ==========================================================
# Helper Functions
# ==========================================================

def check_ollama_connection():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def build_prompt(user_command, old_code):
    parts = []

    parts.append("You are Qwen Coder, an expert software engineer.")
    parts.append("Your job is to generate complete, correct, ready-to-run code.")
    parts.append("")
    parts.append("User requested language:")
    parts.append(language)
    parts.append("")
    parts.append("Code type:")
    parts.append(code_type)
    parts.append("")
    parts.append("Detail level:")
    parts.append(detail_level)
    parts.append("")
    parts.append("User command:")
    parts.append(user_command)
    parts.append("")
    parts.append("Existing code if provided:")
    parts.append(old_code)
    parts.append("")
    parts.append("Instructions:")
    parts.append("1. Understand the user command.")
    parts.append("2. Decide the correct logic.")
    parts.append("3. Generate complete code.")
    parts.append("4. Include all required imports.")
    parts.append("5. Include comments inside the code.")
    parts.append("6. Include error handling where appropriate.")
    parts.append("7. Make the code immediately executable after copy-paste.")
    parts.append("8. If the language is SQL, write clean SQL with comments.")
    parts.append("9. If the language is HTML/CSS, include full HTML document with internal CSS unless user asks separate files.")
    parts.append("10. If the request is JavaScript, include runnable HTML + JS if needed.")
    parts.append("11. If the request is Java, include a complete public class with main method.")
    parts.append("12. If the request is R, include complete R script.")
    parts.append("13. If the request is Python, include complete Python script.")
    parts.append("14. If the request is Streamlit, write complete app.py.")
    parts.append("15. If multiple files are needed, separate them clearly with file names.")
    parts.append("16. Do not use paid APIs.")
    parts.append("17. Return code only. Do not explain outside the code.")

    return "\n".join(parts)


def generate_with_qwen(prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "")


def extract_code(text):
    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def get_extension(selected_language, user_command):
    lang = selected_language.lower()
    cmd = user_command.lower()

    if "python" in lang or "streamlit" in cmd:
        return "py"
    if lang == "r" or " r " in cmd:
        return "R"
    if "sql" in lang or "sql" in cmd:
        return "sql"
    if "html" in lang or "html" in cmd or "css" in cmd:
        return "html"
    if "javascript" in lang or "javascript" in cmd or "js" in cmd:
        return "html"
    if "java" == lang or "java" in cmd:
        return "java"
    if "php" in lang or "php" in cmd:
        return "php"
    if "bash" in lang or "bash" in cmd or "shell" in cmd:
        return "sh"
    if "c++" in lang or "cpp" in cmd:
        return "cpp"

    return "txt"


# ==========================================================
# Buttons
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    generate_button = st.button(
        "🚀 Generate Code",
        use_container_width=True
    )

with col2:
    test_button = st.button(
        "🔍 Test Qwen/Ollama Connection",
        use_container_width=True
    )

# ==========================================================
# Test Connection
# ==========================================================

if test_button:
    if check_ollama_connection():
        st.success("Ollama is running. Qwen model can be used.")
    else:
        st.error("Ollama is not running. Start it using: ollama serve")

# ==========================================================
# Generate Code
# ==========================================================

if generate_button:

    if not command.strip():
        st.warning("Please enter a command.")
        st.stop()

    if not check_ollama_connection():
        st.error("Ollama is not running. Start Ollama first using: ollama serve")
        st.stop()

    prompt = build_prompt(command, existing_code)

    try:
        with st.spinner("Qwen is generating code..."):
            raw_output = generate_with_qwen(prompt)

        final_code = extract_code(raw_output)

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(final_code)

        extension = get_extension(language, command)

        file_name = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + extension
        )

        st.download_button(
            label="⬇️ Download Code",
            data=final_code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("View raw Qwen response"):
            st.write(raw_output)

    except Exception as error:
        st.error("Error generating code: " + str(error))

# ==========================================================
# Bottom information
# ==========================================================

st.divider()

st.subheader("Local Setup")

st.code(
    "ollama serve\nollama pull qwen2.5-coder:7b\nstreamlit run app.py",
    language="bash"
)
