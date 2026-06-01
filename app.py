import streamlit as st
from datetime import datetime
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.set_page_config(
    page_title="No API AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 No-API AI Code Generator")
st.write("This app uses a local Hugging Face Qwen code model. No OpenAI, Groq, Ollama, or API key required.")

st.sidebar.header("Settings")

model_name = st.sidebar.selectbox(
    "Local AI Model",
    [
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    ]
)

language = st.sidebar.selectbox(
    "Programming Language",
    [
        "Python",
        "Streamlit Python",
        "SQL",
        "R",
        "HTML",
        "JavaScript",
        "PHP",
        "Bash"
    ]
)

max_new_tokens = st.sidebar.slider(
    "Maximum Output Length",
    min_value=100,
    max_value=1200,
    value=500,
    step=100
)

temperature = st.sidebar.slider(
    "Creativity",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.1
)


@st.cache_resource
def load_model(selected_model):
    tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(selected_model, trust_remote_code=True)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator


instruction = st.text_area(
    "Enter your coding instruction",
    height=180,
    placeholder="Example: Write Python code to add two numbers"
)

existing_code = st.text_area(
    "Optional: Paste existing code for modification/debugging",
    height=130
)


def build_prompt(user_instruction, old_code):
    parts = []

    parts.append("You are an expert coding assistant.")
    parts.append("Write complete runnable code.")
    parts.append("")
    parts.append("Programming language:")
    parts.append(language)
    parts.append("")
    parts.append("User instruction:")
    parts.append(user_instruction)
    parts.append("")
    parts.append("Existing code:")
    parts.append(old_code)
    parts.append("")
    parts.append("Requirements:")
    parts.append("- Include imports if needed")
    parts.append("- Include comments")
    parts.append("- Include error handling")
    parts.append("- Do not use API keys")
    parts.append("- Return code only")
    parts.append("")
    parts.append("Code:")

    return "\n".join(parts)


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


def template_code(user_instruction):
    lower_text = user_instruction.lower()

    if "add" in lower_text and "number" in lower_text:
        return '''# Python program to add two numbers

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    result = num1 + num2

    print("The sum is:", result)

except ValueError:
    print("Error: Please enter valid numeric values.")
'''

    if "search" in lower_text and "name" in lower_text:
        return '''# Python program to search a name in a string

text = input("Enter a sentence: ")
name = input("Enter name to search: ")

text_lower = text.lower()
name_lower = name.lower()

if name_lower in text_lower:
    count = text_lower.count(name_lower)
    position = text_lower.find(name_lower)

    print("Name found")
    print("Number of occurrences:", count)
    print("First position:", position)
else:
    print("Name not found")
'''

    return ""


def get_extension(lang):
    mapping = {
        "Python": "py",
        "Streamlit Python": "py",
        "SQL": "sql",
        "R": "R",
        "HTML": "html",
        "JavaScript": "js",
        "PHP": "php",
        "Bash": "sh"
    }

    return mapping.get(lang, "txt")


if st.button("🚀 Generate Code", use_container_width=True):

    if not instruction.strip():
        st.warning("Please enter an instruction.")
        st.stop()

    fallback = template_code(instruction)

    try:
        with st.spinner("Loading Qwen model and generating code..."):
            generator = load_model(model_name)

            prompt = build_prompt(instruction, existing_code)

            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )

            raw_text = result[0]["generated_text"]
            generated_code = clean_output(raw_text, prompt)

        if len(generated_code.strip()) < 20 and fallback:
            generated_code = fallback

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(generated_code)

        file_name = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + get_extension(language)
        )

        st.download_button(
            label="⬇️ Download Code",
            data=generated_code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        if fallback:
            st.warning("Model failed, but template code was generated.")
            st.code(fallback)

            st.download_button(
                label="⬇️ Download Template Code",
                data=fallback,
                file_name="generated_code.py",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error("Error: " + str(e))


st.divider()
st.subheader("requirements.txt")
st.code(
    "streamlit\ntransformers\ntorch\naccelerate\nsentencepiece\n",
    language="text"
)

st.subheader("Important")
st.write("This app does not use any API key.")
st.write("On Streamlit Cloud, Qwen model loading may be slow and may fail if memory is limited.")
