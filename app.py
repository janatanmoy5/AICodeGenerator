import streamlit as st
from datetime import datetime
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Qwen No-API Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen No-API Code Generator")
st.write("Enter any coding command. Qwen will generate ready-to-use code.")

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

command = st.text_area(
    "Enter your command",
    height=220,
    placeholder="Example: Write Python code to add two numbers"
)

language = st.selectbox(
    "Programming language",
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
        "C++"
    ]
)

max_tokens = st.slider(
    "Output length",
    min_value=256,
    max_value=2048,
    value=1024,
    step=256
)


@st.cache_resource
def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


def build_prompt(user_command, selected_language):
    prompt = []

    prompt.append("You are an expert software engineer and code generator.")
    prompt.append("The user will give any coding command.")
    prompt.append("You must generate complete, correct, copy-paste-ready code.")
    prompt.append("")
    prompt.append("Programming language:")
    prompt.append(selected_language)
    prompt.append("")
    prompt.append("User command:")
    prompt.append(user_command)
    prompt.append("")
    prompt.append("Rules:")
    prompt.append("1. Understand the command.")
    prompt.append("2. Create the correct programming logic.")
    prompt.append("3. Generate full runnable code.")
    prompt.append("4. Include imports when needed.")
    prompt.append("5. Include helpful comments inside the code.")
    prompt.append("6. Include input and output handling where useful.")
    prompt.append("7. Do not use API keys.")
    prompt.append("8. Do not write explanation outside the code.")
    prompt.append("9. Return only the final code.")
    prompt.append("10. If Streamlit is requested, generate a complete app.py.")
    prompt.append("11. If HTML/CSS/JavaScript is requested, generate a full HTML file.")
    prompt.append("12. If Java is requested, include a complete class with main method.")
    prompt.append("13. If SQL is requested, generate clean SQL with comments.")
    prompt.append("")
    prompt.append("Final code:")

    return "\n".join(prompt)


def generate_code(user_command, selected_language, token_limit):
    tokenizer, model = load_qwen()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Generate only complete runnable code."
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
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "Final code:" in generated:
        generated = generated.split("Final code:")[-1].strip()

    code_blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        generated,
        re.DOTALL
    )

    if code_blocks:
        generated = "\n\n".join(code_blocks).strip()

    return generated.strip()


def get_file_extension(selected_language, user_command):
    text = (selected_language + " " + user_command).lower()

    if "python" in text or "streamlit" in text:
        return "py"
    if "sql" in text:
        return "sql"
    if "html" in text or "css" in text or "javascript" in text:
        return "html"
    if "java" in text:
        return "java"
    if " r " in text or selected_language == "R":
        return "R"
    if "php" in text:
        return "php"
    if "bash" in text:
        return "sh"
    if "c++" in text:
        return "cpp"

    return "txt"


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter a command.")
        st.stop()

    try:
        with st.spinner("Loading Qwen and generating code..."):
            code = generate_code(
                command,
                language,
                max_tokens
            )

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(code)

        extension = get_file_extension(language, command)

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

    except Exception as e:
        st.error("Qwen model could not run on this Streamlit Cloud machine.")
        st.error(str(e))
