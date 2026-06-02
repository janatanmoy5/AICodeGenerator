import os
import re
import gc
import traceback
from datetime import datetime

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/huggingface"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(
    page_title="Qwen Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen Code Generator")
st.write("User gives any coding instruction. Qwen model generates complete code.")

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

with st.sidebar:
    st.header("Model Settings")

    language = st.selectbox(
        "Programming Language",
        [
            "Auto Detect",
            "Python",
            "R",
            "Perl",
            "C",
            "C++",
            "Java",
            "JavaScript",
            "TypeScript",
            "HTML",
            "CSS",
            "HTML + CSS + JavaScript",
            "SQL",
            "PHP",
            "Bash",
            "Go",
            "Rust"
        ]
    )

    max_tokens = st.slider(
        "Output Length",
        min_value=128,
        max_value=1024,
        value=512,
        step=128
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    show_debug = st.checkbox("Show Debug Error", value=True)

    if st.button("Clear Model Cache"):
        st.cache_resource.clear()
        gc.collect()
        st.success("Model cache cleared.")


command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: Write R code to add two numbers"
)


@st.cache_resource(show_spinner="Loading Qwen model...")
def load_qwen_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


def detect_language(user_command, selected_language):
    text = user_command.lower()

    if selected_language != "Auto Detect":
        return selected_language

    if "r code" in text or text.startswith("r "):
        return "R"
    if "perl" in text:
        return "Perl"
    if "c++" in text or "cpp" in text:
        return "C++"
    if " c " in f" {text} ":
        return "C"
    if "java" in text and "javascript" not in text:
        return "Java"
    if "javascript" in text or " js " in f" {text} ":
        return "JavaScript"
    if "typescript" in text:
        return "TypeScript"
    if "html" in text:
        return "HTML"
    if "css" in text:
        return "CSS"
    if "sql" in text:
        return "SQL"
    if "php" in text:
        return "PHP"
    if "bash" in text or "shell" in text:
        return "Bash"
    if "go " in f" {text} ":
        return "Go"
    if "rust" in text:
        return "Rust"

    return "Python"


def build_prompt(user_command, selected_language):
    detected_language = detect_language(user_command, selected_language)

    prompt = f"""
You are Qwen Coder, a professional code generation model.

Your task:
Generate complete, correct, runnable code.

Programming language:
{detected_language}

User instruction:
{user_command}

Rules:
1. Generate code only.
2. Do not explain outside the code.
3. Use only the requested programming language.
4. If the user asks for R, generate R code only.
5. If the user asks for Python, generate Python code only.
6. If the user asks for SQL, generate SQL only.
7. If the user asks for HTML/CSS/JavaScript, generate a complete HTML file if useful.
8. If the user asks for Java, include public class Main.
9. If the user asks for Perl, include use strict and use warnings.
10. Code must be copy-paste-ready.
11. Include imports or libraries if needed.
12. Include comments inside the code.

Return final code only.
"""

    return prompt


def clean_output(text):
    # Remove markdown code blocks if Qwen returns them
    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    # Remove repeated prompt if present
    if "Return final code only." in text:
        text = text.split("Return final code only.")[-1].strip()

    return text.strip()


def generate_code_with_qwen(user_command, selected_language, token_limit, temp):
    tokenizer, model = load_qwen_model()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Generate only final runnable code."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=True,
            temperature=max(temp, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_output(decoded)


def get_extension(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    mapping = {
        "Python": "py",
        "R": "R",
        "Perl": "pl",
        "C": "c",
        "C++": "cpp",
        "Java": "java",
        "JavaScript": "js",
        "TypeScript": "ts",
        "HTML": "html",
        "CSS": "css",
        "HTML + CSS + JavaScript": "html",
        "SQL": "sql",
        "PHP": "php",
        "Bash": "sh",
        "Go": "go",
        "Rust": "rs"
    }

    return mapping.get(lang, "txt")


if st.button("🚀 Generate Code with Qwen", use_container_width=True):

    if not command.strip():
        st.warning("Please enter your coding instruction.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    try:
        status.info("Step 1: Loading Qwen pretrained model...")
        progress.progress(25)

        tokenizer, model = load_qwen_model()

        status.info("Step 2: Sending your instruction to Qwen...")
        progress.progress(50)

        status.info("Step 3: Generating code...")
        progress.progress(75)

        code = generate_code_with_qwen(
            user_command=command,
            selected_language=language,
            token_limit=max_tokens,
            temp=temperature
        )

        progress.progress(100)
        status.success("Qwen generated the code successfully.")

        st.subheader("Generated Code")
        st.code(code)

        extension = get_extension(command, language)

        file_name = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + extension
        )

        st.download_button(
            label="⬇️ Download Code",
            data=code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

    except Exception as error:
        status.error("Qwen model failed to generate code.")

        st.error("The model did not run successfully on this server.")

        if show_debug:
            st.subheader("Debug Traceback")
            st.code(traceback.format_exc())
