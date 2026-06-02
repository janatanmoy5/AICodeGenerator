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
    page_title="Lightweight AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Lightweight AI Code Generator")
st.write("User instruction → lightweight pretrained code model → generated code.")

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Salesforce/codegen-350M-mono"
)

with st.sidebar:
    st.header("Model Settings")

    language = st.selectbox(
        "Programming Language",
        [
            "Auto Detect",
            "Python",
            "R",
            "SQL",
            "Java",
            "C",
            "C++",
            "Perl",
            "HTML",
            "CSS",
            "JavaScript",
            "PHP",
            "Bash"
        ]
    )

    max_tokens = st.slider(
        "Output Length",
        min_value=32,
        max_value=256,
        value=128,
        step=32
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


@st.cache_resource(show_spinner="Loading lightweight code model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


def detect_language(user_command, selected_language):
    text = user_command.lower().strip()

    if selected_language != "Auto Detect":
        return selected_language

    if "r code" in text or text.startswith("r "):
        return "R"
    if "sql" in text:
        return "SQL"
    if "java" in text and "javascript" not in text:
        return "Java"
    if "javascript" in text:
        return "JavaScript"
    if "html" in text:
        return "HTML"
    if "css" in text:
        return "CSS"
    if "c++" in text or "cpp" in text:
        return "C++"
    if " c " in f" {text} ":
        return "C"
    if "perl" in text:
        return "Perl"
    if "php" in text:
        return "PHP"
    if "bash" in text or "shell" in text:
        return "Bash"

    return "Python"


def build_prompt(user_command, selected_language):
    lang = detect_language(
        user_command,
        selected_language
    )

    if lang == "R":
        prefix = "# Write complete R code only\n"
    elif lang == "SQL":
        prefix = "-- Write complete SQL code only\n"
    elif lang == "Java":
        prefix = "// Write complete Java code only\n"
    elif lang == "C++":
        prefix = "// Write complete C++ code only\n"
    elif lang == "C":
        prefix = "// Write complete C code only\n"
    elif lang == "Perl":
        prefix = "# Write complete Perl code only\n"
    elif lang == "HTML":
        prefix = "<!-- Write complete HTML code only -->\n"
    elif lang == "CSS":
        prefix = "/* Write complete CSS code only */\n"
    elif lang == "JavaScript":
        prefix = "// Write complete JavaScript code only\n"
    elif lang == "PHP":
        prefix = "<?php\n// Write complete PHP code only\n"
    elif lang == "Bash":
        prefix = "#!/bin/bash\n# Write complete Bash code only\n"
    else:
        prefix = "# Write complete Python code only\n"

    prompt = (
        prefix
        + "# User instruction:\n"
        + user_command
        + "\n\n"
        + "# Code:\n"
    )

    return prompt


def clean_output(text, prompt):
    text = text.replace(prompt, "")

    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_code(user_command, selected_language, token_limit, temp):
    tokenizer, model = load_model()

    prompt = build_prompt(
        user_command,
        selected_language
    )

    inputs = tokenizer(
        prompt,
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

    return clean_output(
        decoded,
        prompt
    )


def fallback_code(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

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
        return '''-- Basic SQL code

SELECT *
FROM table_name;
'''

    if lang == "Java":
        return '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from Java");
    }
}
'''

    if lang == "HTML":
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Generated Page</title>
</head>
<body>
    <h1>Hello from HTML</h1>
</body>
</html>
'''

    return f'''# Fallback code

# Detected language: {lang}
# User request:
# {user_command}
'''


def get_extension(user_command, selected_language):
    lang = detect_language(
        user_command,
        selected_language
    )

    mapping = {
        "Python": "py",
        "R": "R",
        "SQL": "sql",
        "Java": "java",
        "C": "c",
        "C++": "cpp",
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
        st.warning("Please enter your coding instruction.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    try:
        if not MODEL_READY:
            status.error("Model dependencies are missing.")
            code = fallback_code(
                command,
                language
            )

            if show_debug:
                st.subheader("Debug Traceback")
                st.code(IMPORT_ERROR)

        else:
            status.info("Step 1: Loading lightweight pretrained model...")
            progress.progress(30)

            load_model()

            status.info("Step 2: Generating code...")
            progress.progress(70)

            code = generate_code(
                user_command=command,
                selected_language=language,
                token_limit=max_tokens,
                temp=temperature
            )

            if not code.strip():
                code = fallback_code(
                    command,
                    language
                )

            progress.progress(100)
            status.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(code)

        extension = get_extension(
            command,
            language
        )

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

    except Exception:
        status.error("Model failed. Showing fallback code.")

        code = fallback_code(
            command,
            language
        )

        st.subheader("Generated Code")
        st.code(code)

        if show_debug:
            st.subheader("Debug Traceback")
            st.code(traceback.format_exc())
