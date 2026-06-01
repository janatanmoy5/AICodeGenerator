import streamlit as st
from datetime import datetime
import re

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception as e:
    st.error("Transformers is not installed. Please check requirements.txt.")
    st.stop()


st.set_page_config(
    page_title="No API Qwen Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 No API Qwen Code Generator")
st.write("Generate code without OpenAI, Groq, or API key. This app uses a Qwen model from Hugging Face.")

st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Choose Qwen Model",
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
        "Flask Python",
        "SQL",
        "R",
        "HTML",
        "JavaScript",
        "PHP",
        "Bash"
    ]
)

max_new_tokens = st.sidebar.slider(
    "Output Length",
    min_value=100,
    max_value=1500,
    value=600,
    step=100
)

temperature = st.sidebar.slider(
    "Creativity",
    min_value=0.1,
    max_value=1.0,
    value=0.2,
    step=0.1
)


@st.cache_resource
def load_qwen_model(selected_model):
    tokenizer = AutoTokenizer.from_pretrained(
        selected_model,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        trust_remote_code=True
    )

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return text_generator


instruction = st.text_area(
    "Enter your coding command",
    height=180,
    placeholder="Example: Write Python code to add two numbers"
)

existing_code = st.text_area(
    "Optional: Paste existing code for debugging or modification",
    height=140
)


def build_prompt(user_instruction, old_code):
    parts = []
    parts.append("You are an expert academic coding assistant.")
    parts.append("Write complete runnable code only.")
    parts.append("")
    parts.append("Programming language:")
    parts.append(language)
    parts.append("")
    parts.append("User command:")
    parts.append(user_instruction)
    parts.append("")
    parts.append("Existing code:")
    parts.append(old_code)
    parts.append("")
    parts.append("Rules:")
    parts.append("1. Write complete executable code.")
    parts.append("2. Include all imports.")
    parts.append("3. Include comments.")
    parts.append("4. Include error handling.")
    parts.append("5. Do not use API keys.")
    parts.append("6. Do not write explanation.")
    parts.append("7. Return code only.")
    parts.append("")
    parts.append("Code:")

    return "\n".join(parts)


def clean_generated_text(text, prompt):
    text = text.replace(prompt, "")

    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def template_fallback(user_instruction):
    text = user_instruction.lower()

    if "add" in text and "number" in text:
        return '''# Python program to add two numbers

try:
    number1 = float(input("Enter first number: "))
    number2 = float(input("Enter second number: "))

    total = number1 + number2

    print("First number:", number1)
    print("Second number:", number2)
    print("Sum:", total)

except ValueError:
    print("Error: Please enter valid numeric values.")
'''

    if "search" in text and "name" in text:
        return '''# Python program to search a name in a string

try:
    sentence = input("Enter a sentence: ")
    name = input("Enter name to search: ")

    sentence_lower = sentence.lower()
    name_lower = name.lower()

    if name_lower in sentence_lower:
        count = sentence_lower.count(name_lower)
        first_position = sentence_lower.find(name_lower)

        print("Name found:", name)
        print("Occurrences:", count)
        print("First position:", first_position)
    else:
        print("Name not found:", name)

except Exception as e:
    print("Error:", e)
'''

    if "csv" in text:
        return '''# Python program to read a CSV file

import pandas as pd

try:
    file_path = input("Enter CSV file path: ")

    df = pd.read_csv(file_path)

    print("CSV loaded successfully")
    print("Shape:", df.shape)
    print(df.head())

except FileNotFoundError:
    print("Error: File not found.")

except Exception as e:
    print("Error:", e)
'''

    if "streamlit" in text:
        return '''# Streamlit starter app

import streamlit as st

st.title("My Streamlit App")

user_input = st.text_input("Enter something:")

if st.button("Submit"):
    st.write("You entered:", user_input)
'''

    if "sql" in text:
        return '''-- SQL query example

SELECT
    *
FROM
    table_name
WHERE
    condition_column = 'value';
'''

    return ""


def get_file_extension(selected_language):
    mapping = {
        "Python": "py",
        "Streamlit Python": "py",
        "Flask Python": "py",
        "SQL": "sql",
        "R": "R",
        "HTML": "html",
        "JavaScript": "js",
        "PHP": "php",
        "Bash": "sh"
    }

    return mapping.get(selected_language, "txt")


if st.button("🚀 Generate Code", use_container_width=True):

    if not instruction.strip():
        st.warning("Please enter a coding command.")
        st.stop()

    fallback_code = template_fallback(instruction)

    try:
        with st.spinner("Loading Qwen model and generating code..."):
            generator = load_qwen_model(model_name)

            prompt = build_prompt(instruction, existing_code)

            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )

            raw_output = result[0]["generated_text"]

            generated_code = clean_generated_text(raw_output, prompt)

        if len(generated_code.strip()) < 20 and fallback_code:
            generated_code = fallback_code

        st.success("Code generated successfully.")

        st.subheader("Generated Code")
        st.code(generated_code)

        file_name = (
            "generated_code_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + get_file_extension(language)
        )

        st.download_button(
            label="⬇️ Download Code",
            data=generated_code,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("Raw model output"):
            st.write(raw_output)

    except Exception as e:
        if fallback_code:
            st.warning("Qwen model failed, but template code was generated.")
            st.code(fallback_code)

            st.download_button(
                label="⬇️ Download Template Code",
                data=fallback_code,
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

st.subheader("Run Command")
st.code(
    "streamlit run app.py",
    language="bash"
)

st.warning(
    "Note: This uses a local Hugging Face model. On Streamlit Cloud, it may be slow or fail if memory is not enough. "
    "For best result, use Qwen/Qwen2.5-Coder-0.5B-Instruct first."
)
