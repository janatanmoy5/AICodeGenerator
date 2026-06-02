import os
import re
import gc
from datetime import datetime

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/huggingface"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(
    page_title="Qwen Local Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen Local Pretrained Code Generator")
st.write("This app uses a pretrained Qwen model locally. No API call is used.")

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

with st.sidebar:
    st.header("Settings")

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
        128,
        1024,
        512,
        128
    )

    temperature = st.slider(
        "Creativity",
        0.0,
        1.0,
        0.2,
        0.1
    )

    if st.button("Clear Memory"):
        st.cache_resource.clear()
        gc.collect()
        st.success("Memory cleared.")


command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: Write R code to add two numbers"
)


@st.cache_resource(show_spinner="Loading pretrained Qwen model...")
def load_model():
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


def build_prompt(user_command, selected_language):
    return f"""
You are Qwen Coder, an expert programming assistant.

Generate complete, runnable, copy-paste-ready code.

Selected language:
{selected_language}

User instruction:
{user_command}

Rules:
1. Return only code.
2. No explanation outside code.
3. Use the requested programming language only.
4. If Auto Detect is selected, infer the language from the instruction.
5. Support Python, R, Perl, C, C++, Java, JavaScript, TypeScript, HTML, CSS, SQL, PHP, Bash, Go, Rust.
6. For Java, use public class Main.
7. For R, use valid R syntax only.
8. For Perl, include use strict and use warnings.
9. For HTML/CSS/JavaScript, return a complete HTML file when useful.
10. Include imports and comments when needed.

Final code:
"""


def clean_output(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_code(user_command, selected_language):
    tokenizer, model = load_model()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Return only runnable code."
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
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_output(decoded)


def get_extension(selected_language, user_command):
    text = f"{selected_language} {user_command}".lower()

    if "python" in text or "streamlit" in text:
        return "py"
    if "r code" in text or selected_language == "R":
        return "R"
    if "perl" in text:
        return "pl"
    if "c++" in text or "cpp" in text:
        return "cpp"
    if " c " in f" {text} ":
        return "c"
    if "java" in text and "javascript" not in text:
        return "java"
    if "javascript" in text:
        return "js"
    if "typescript" in text:
        return "ts"
    if "html" in text or "css" in text:
        return "html"
    if "sql" in text:
        return "sql"
    if "php" in text:
        return "php"
    if "bash" in text or "shell" in text:
        return "sh"
    if "go" in text:
        return "go"
    if "rust" in text:
        return "rs"

    return "txt"


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter your coding instruction.")
        st.stop()

    try:
        with st.spinner("Qwen is generating code locally..."):
            code = generate_code(command, language)

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(code)

        ext = get_extension(language, command)

        file_name = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + ext
        )

        st.download_button(
            label="⬇️ Download Code",
            data=code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        st.error("Local pretrained Qwen model could not run.")
        st.error(str(e))
