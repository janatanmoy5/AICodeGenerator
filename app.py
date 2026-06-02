import os
import re
import gc
import traceback
from datetime import datetime

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/huggingface"

import streamlit as st

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODEL_READY = True
    IMPORT_ERROR = ""
except Exception:
    MODEL_READY = False
    IMPORT_ERROR = traceback.format_exc()


st.set_page_config(
    page_title="Qwen Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen Code Generator")
st.write("User instruction → Qwen model → generated code.")

MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

with st.sidebar:
    language = st.selectbox(
        "Language",
        [
            "Auto Detect", "Python", "R", "SQL", "Java", "C++",
            "C", "Perl", "HTML", "CSS", "JavaScript", "PHP", "Bash"
        ]
    )

    max_tokens = st.slider(
        "Output length",
        32,
        256,
        128,
        32
    )

    temperature = st.slider(
        "Creativity",
        0.0,
        1.0,
        0.2,
        0.1
    )

    show_debug = st.checkbox("Show debug", value=True)

    if st.button("Clear memory"):
        st.cache_resource.clear()
        gc.collect()
        st.success("Memory cleared")


command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: Write R code to add two numbers"
)


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model.eval()

    return tokenizer, model


def detect_language(user_command, selected_language):
    text = user_command.lower()

    if selected_language != "Auto Detect":
        return selected_language

    if "r code" in text or text.startswith("r "):
        return "R"
    if "sql" in text:
        return "SQL"
    if "java" in text and "javascript" not in text:
        return "Java"
    if "c++" in text or "cpp" in text:
        return "C++"
    if "perl" in text:
        return "Perl"
    if "html" in text:
        return "HTML"
    if "css" in text:
        return "CSS"
    if "javascript" in text or " js " in f" {text} ":
        return "JavaScript"
    if "php" in text:
        return "PHP"
    if "bash" in text or "shell" in text:
        return "Bash"

    return "Python"


def build_prompt(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    return f"""
You are Qwen Coder.

Generate complete runnable code only.

Language:
{lang}

User instruction:
{user_command}

Rules:
- Return only code.
- No explanation.
- Use only {lang}.
- If R, write valid R syntax.
- If Java, include public class Main.
- If Perl, include use strict and use warnings.
- If HTML/CSS/JavaScript, generate full HTML if useful.
"""


def clean_output(text):
    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    if "assistant" in text:
        parts = text.split("assistant")
        text = parts[-1]

    return text.strip()


def generate_with_qwen(user_command, selected_language, token_limit, temp):
    tokenizer, model = load_model()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Return only final code."
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


def fallback_code(user_command, selected_language):
    lang = detect_language(user_command, selected_language)
    text = user_command.lower()

    if "add" in text and "number" in text:
        if lang == "R":
            return '''# R program to add two numbers

a <- as.numeric(readline("Enter first number: "))
b <- as.numeric(readline("Enter second number: "))

sum_result <- a + b

cat("Sum:", sum_result, "\\n")
'''

        if lang == "Python":
            return '''# Python program to add two numbers

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

print("Sum:", a + b)
'''

        if lang == "Java":
            return '''import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter first number: ");
        double a = sc.nextDouble();

        System.out.print("Enter second number: ");
        double b = sc.nextDouble();

        System.out.println("Sum: " + (a + b));

        sc.close();
    }
}
'''

    if lang == "R":
        return '''# Basic R code

message <- "Hello from R"
print(message)
'''

    if lang == "Python":
        return '''# Basic Python code

print("Hello from Python")
'''

    if lang == "SQL":
        return '''-- Basic SQL query

SELECT *
FROM table_name;
'''

    if lang == "HTML":
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Basic Page</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>
'''

    return f'''# Fallback code

# Qwen did not complete on this server.
# Detected language: {lang}
# Request: {user_command}
'''


def get_extension(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    mapping = {
        "Python": "py",
        "R": "R",
        "SQL": "sql",
        "Java": "java",
        "C++": "cpp",
        "C": "c",
        "Perl": "pl",
        "HTML": "html",
        "CSS": "css",
        "JavaScript": "js",
        "PHP": "php",
        "Bash": "sh"
    }

    return mapping.get(lang, "txt")


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter instruction.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    code = ""

    if not MODEL_READY:
        status.error("Torch / Transformers not installed.")
        code = fallback_code(command, language)

        if show_debug:
            st.code(IMPORT_ERROR)

    else:
        try:
            status.info("Step 1: Loading Qwen model...")
            progress.progress(25)

            load_model()

            status.info("Step 2: Generating code...")
            progress.progress(60)

            code = generate_with_qwen(
                command,
                language,
                max_tokens,
                temperature
            )

            progress.progress(100)
            status.success("Qwen generated code.")

        except Exception:
            status.error("Qwen failed. Showing fallback code.")
            code = fallback_code(command, language)

            if show_debug:
                st.subheader("Debug")
                st.code(traceback.format_exc())

    st.subheader("Generated Code")
    st.code(code)

    filename = (
        "generated_code_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + get_extension(command, language)
    )

    st.download_button(
        "⬇️ Download Code",
        data=code,
        file_name=filename,
        mime="text/plain",
        use_container_width=True
    )
