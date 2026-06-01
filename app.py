import streamlit as st
from datetime import datetime
import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


st.set_page_config(
    page_title="AI Project Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Project Code Generator")
st.write("Enter any command. The app generates project code and can create plots instantly.")

with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Qwen Model",
        [
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        ]
    )

    language = st.selectbox(
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

    max_new_tokens = st.slider(
        "Output Length",
        min_value=200,
        max_value=2000,
        value=900,
        step=100
    )

    temperature = st.slider(
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

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator


command = st.text_area(
    "Enter your command",
    height=220,
    placeholder="Example: Write Python code to add two numbers and generate a bar plot"
)

uploaded_file = st.file_uploader(
    "Optional: Upload CSV file for instant plot",
    type=["csv"]
)


def build_prompt(user_command):
    parts = []
    parts.append("You are an expert software engineer.")
    parts.append("Generate complete project-ready executable code.")
    parts.append("")
    parts.append("Language:")
    parts.append(language)
    parts.append("")
    parts.append("User command:")
    parts.append(user_command)
    parts.append("")
    parts.append("Rules:")
    parts.append("1. Generate complete runnable code.")
    parts.append("2. Include all required imports.")
    parts.append("3. Include complete logic.")
    parts.append("4. Include comments.")
    parts.append("5. Include error handling.")
    parts.append("6. Do not use any API key.")
    parts.append("7. If plotting is requested, include matplotlib plot code.")
    parts.append("8. If Streamlit app is requested, output complete app.py.")
    parts.append("9. Do not explain. Return code only.")
    parts.append("")
    parts.append("Project code:")
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


def fallback_code(user_command):
    text = user_command.lower()

    if "add" in text and "number" in text:
        return '''# Python program to add two numbers and plot result

import matplotlib.pyplot as plt

def main():
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        total = num1 + num2

        print("First number:", num1)
        print("Second number:", num2)
        print("Sum:", total)

        labels = ["Number 1", "Number 2", "Sum"]
        values = [num1, num2, total]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.title("Addition Result")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("Error: Please enter valid numeric values.")

if __name__ == "__main__":
    main()
'''

    if "search" in text and "name" in text:
        return '''# Python program to search a name in a string

def main():
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

    except Exception as error:
        print("Error:", error)

if __name__ == "__main__":
    main()
'''

    if "pca" in text:
        return '''# PCA plot from CSV file

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    file_path = input("Enter CSV file path: ")

    try:
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.empty:
            print("No numeric columns found.")
            return

        scaled = StandardScaler().fit_transform(numeric_df)

        pca = PCA(n_components=2)
        result = pca.fit_transform(scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(result[:, 0], result[:, 1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Plot")
        plt.tight_layout()
        plt.show()

        print("Explained variance:", pca.explained_variance_ratio_)

    except Exception as error:
        print("Error:", error)

if __name__ == "__main__":
    main()
'''

    if "csv" in text or "plot" in text:
        return '''# Python CSV plotting program

import pandas as pd
import matplotlib.pyplot as plt

def main():
    file_path = input("Enter CSV file path: ")

    try:
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.empty:
            print("No numeric columns found.")
            return

        numeric_df.hist(figsize=(10, 8))
        plt.tight_layout()
        plt.show()

        print("CSV shape:", df.shape)
        print(df.head())

    except Exception as error:
        print("Error:", error)

if __name__ == "__main__":
    main()
'''

    if "streamlit" in text:
        return '''# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Plot App", layout="wide")

st.title("Data Plot App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully")
        st.dataframe(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if numeric_cols:
            column = st.selectbox("Select numeric column", numeric_cols)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[column].dropna(), bins=20)
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found.")

    except Exception as error:
        st.error(f"Error: {error}")
else:
    st.info("Please upload a CSV file.")
'''

    return '''# Python starter code generated from command

def main():
    print("Command received.")
    print("Please give a more specific command such as:")
    print("Write Python code to add two numbers")
    print("Create PCA plot from CSV")
    print("Create Streamlit app for CSV plotting")

if __name__ == "__main__":
    main()
'''


def get_extension(selected_language):
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


def create_instant_plot(user_command, csv_file):
    text = user_command.lower()

    if csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(
            {
                "Sample": ["A", "B", "C", "D", "E"],
                "Value1": [10, 25, 15, 35, 30],
                "Value2": [5, 15, 20, 25, 18]
            }
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for plotting.")
        return

    st.subheader("Instant Plot Output")

    if "line" in text:
        y_col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df[y_col], marker="o")
        ax.set_title("Line Plot")
        ax.set_xlabel("Index")
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    elif "scatter" in text and len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x_col], df[y_col])
        ax.set_title("Scatter Plot")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    elif "hist" in text or "histogram" in text:
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title("Histogram")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif "pca" in text and len(numeric_cols) >= 2:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        numeric_df = df[numeric_cols].dropna()

        scaled = StandardScaler().fit_transform(numeric_df)

        pca = PCA(n_components=2)
        result = pca.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(result[:, 0], result[:, 1])
        ax.set_title("PCA Plot")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        variance_df = pd.DataFrame(
            {
                "Component": ["PC1", "PC2"],
                "Explained Variance": pca.explained_variance_ratio_
            }
        )

        st.dataframe(variance_df)

    else:
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(8, 5))

        if "Sample" in df.columns:
            ax.bar(df["Sample"].astype(str), df[col])
            ax.set_xlabel("Sample")
        else:
            ax.bar(df.index.astype(str), df[col])
            ax.set_xlabel("Index")

        ax.set_title("Bar Plot")
        ax.set_ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)


if st.button("🚀 Generate Code and Plot", use_container_width=True):

    if not command.strip():
        st.warning("Please enter a command.")
        st.stop()

    prompt = build_prompt(command)
    backup_code = fallback_code(command)

    generated_code = backup_code
    raw_output = ""

    if TRANSFORMERS_AVAILABLE:
        try:
            with st.spinner("Generating code using Qwen model..."):
                generator = load_qwen_model(model_name)

                result = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1
                )

                raw_output = result[0]["generated_text"]
                generated_code = clean_output(raw_output, prompt)

                if len(generated_code.strip()) < 30:
                    generated_code = backup_code

        except Exception:
            generated_code = backup_code
    else:
        generated_code = backup_code

    st.success("Project output generated.")

    st.subheader("Project Code Output")
    st.code(generated_code)

    file_name = (
        "project_output_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + get_extension(language)
    )

    st.download_button(
        label="⬇️ Download Project Code",
        data=generated_code,
        file_name=file_name,
        mime="text/plain",
        use_container_width=True
    )

    if (
        "plot" in command.lower()
        or "graph" in command.lower()
        or "chart" in command.lower()
        or "pca" in command.lower()
        or "hist" in command.lower()
        or "scatter" in command.lower()
        or "line" in command.lower()
        or "bar" in command.lower()
    ):
        try:
            create_instant_plot(command, uploaded_file)
        except Exception as error:
            st.error("Plot error: " + str(error))
