import os
import re
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")

import streamlit as st

st.set_page_config(
    page_title="Basic Qwen Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Basic Qwen Code Generator")
st.write("Enter any coding instruction. Qwen will generate code.")

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

language = st.selectbox(
    "Language",
    [
        "Auto Detect",
        "Python",
        "R",
        "PHP",
        "HTML",
        "CSS",
        "JavaScript",
        "SQL",
        "Java",
        "C++",
        "Perl",
        "Bash"
    ]
)

instruction = st.text_area(
    "Enter your coding instruction",
    height=180,
    placeholder="Example: write html code to print my name is Tanmoy"
)

max_tokens = st.slider(
    "Output length",
    min_value=64,
    max_value=384,
    value=192,
    step=64
)


@st.cache_resource(show_spinner="Loading small Qwen model...")
def load_model():
    import torch
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )

    model.eval()

    return tokenizer, model


def build_prompt(user_instruction, selected_language):
    return f"""
You are a coding assistant.

Write complete runnable code only.

Language:
{selected_language}

User instruction:
{user_instruction}

Rules:
- Return only code.
- No explanation.
- Code must be copy-paste ready.
- Support Python, R, PHP, HTML, CSS, JavaScript, SQL, Java, C++, Perl, Bash.
- If Auto Detect, infer the language from the instruction.
- For HTML/CSS/JavaScript, create full HTML when useful.
- For Java, use public class Main.
- For Perl, include use strict and use warnings.

Code:
"""


def clean_output(text):
    if "Code:" in text:
        text = text.split("Code:")[-1].strip()

    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_code(user_instruction, selected_language, token_limit):
    import torch

    tokenizer, model = load_model()

    prompt = build_prompt(user_instruction, selected_language)

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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_output(result)


def get_extension(selected_language, user_instruction):
    text = (selected_language + " " + user_instruction).lower()

    if "python" in text:
        return "py"
    if " r " in f" {text} " or selected_language == "R":
        return "R"
    if "php" in text:
        return "php"
    if "html" in text or "css" in text or "javascript" in text:
        return "html"
    if "sql" in text:
        return "sql"
    if "java" in text and "javascript" not in text:
        return "java"
    if "c++" in text:
        return "cpp"
    if "perl" in text:
        return "pl"
    if "bash" in text:
        return "sh"

    return "txt"


if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

st.subheader("Code Output")

output_box = st.empty()

if st.session_state.generated_code:
    output_box.code(st.session_state.generated_code)
else:
    output_box.code("# Generated code will appear here.")


if st.button("🚀 Generate Code", use_container_width=True):
    if not instruction.strip():
        st.warning("Please enter an instruction.")
        st.stop()

    try:
        with st.status("Qwen is generating code...", expanded=True) as status:
            st.write("Loading small Qwen model...")
            st.write("Preparing instruction...")
            st.write("Generating code...")

            code = generate_code(
                user_instruction=instruction,
                selected_language=language,
                token_limit=max_tokens
            )

            st.session_state.generated_code = code
            output_box.code(code)

            status.update(
                label="Code generated successfully.",
                state="complete"
            )

    except Exception as error:
        st.error("Qwen could not run on this server. Render may not have enough RAM.")
        st.exception(error)


if st.session_state.generated_code:
    filename = (
        "generated_code_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + get_extension(language, instruction)
    )

    st.download_button(
        label="⬇️ Download Code",
        data=st.session_state.generated_code,
        file_name=filename,
        mime="text/plain",
        use_container_width=True
    )
