import os
import re
import gc
from datetime import datetime

import streamlit as st

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    QWEN_READY = True
except Exception as import_error:
    QWEN_READY = False
    IMPORT_ERROR = import_error


st.set_page_config(
    page_title="Qwen AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen AI Code Generator")
st.write("Write any coding command. Qwen will generate ready-to-copy code.")

MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

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
        min_value=128,
        max_value=1536,
        value=768,
        step=128
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    show_debug = st.checkbox("Show Debug Info", value=True)

    if st.button("Clear Model Cache"):
        st.cache_resource.clear()
        gc.collect()
        st.success("Cache cleared.")


command = st.text_area(
    "Enter your coding command",
    height=220,
    placeholder="Example: Write R code to add two numbers"
)


@st.cache_resource(show_spinner=False)
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


def build_prompt(user_command, selected_language):
    parts = []

    parts.append("You are Qwen Coder, an expert programming assistant.")
    parts.append("Generate complete, correct, copy-paste-ready code.")
    parts.append("")
    parts.append("Selected language:")
    parts.append(selected_language)
    parts.append("")
    parts.append("User command:")
    parts.append(user_command)
    parts.append("")
    parts.append("Rules:")
    parts.append("1. Generate code in the requested language.")
    parts.append("2. If Auto Detect is selected, infer language from the command.")
    parts.append("3. Support Python, R, Perl, C, C++, Java, JavaScript, HTML, CSS, SQL, PHP, Bash, Go, Rust.")
    parts.append("4. Code must be runnable after copy-paste.")
    parts.append("5. Include imports or package calls if required.")
    parts.append("6. Include comments inside code.")
    parts.append("7. Do not explain outside code.")
    parts.append("8. Do not use API keys.")
    parts.append("9. For Java, use public class Main.")
    parts.append("10. For R, use valid R syntax only.")
    parts.append("11. For Perl, include use strict and use warnings.")
    parts.append("12. For HTML/CSS/JavaScript, create a full HTML document if useful.")
    parts.append("13. Return only final code.")
    parts.append("")
    parts.append("Final code:")

    return "\n".join(parts)


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


def generate_with_qwen(user_command, selected_language, token_limit, temp):
    tokenizer, model = load_qwen_model()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Return only complete runnable code."
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

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_output(result)


def fallback_code(user_command, selected_language):
    text = user_command.lower()
    lang = selected_language.lower()

    if selected_language == "Auto Detect":
        if "r code" in text or text.startswith("r "):
            lang = "r"
        elif "perl" in text:
            lang = "perl"
        elif "c++" in text or "cpp" in text:
            lang = "c++"
        elif "java" in text and "javascript" not in text:
            lang = "java"
        elif "javascript" in text:
            lang = "javascript"
        elif "html" in text:
            lang = "html"
        elif "css" in text:
            lang = "css"
        elif "sql" in text:
            lang = "sql"
        elif "php" in text:
            lang = "php"
        elif "bash" in text or "shell" in text:
            lang = "bash"
        else:
            lang = "python"

    if "add" in text and "number" in text:
        if lang == "r":
            return """# R program to add two numbers

add_two_numbers <- function(a, b) {
  return(a + b)
}

num1 <- as.numeric(readline(prompt = "Enter first number: "))
num2 <- as.numeric(readline(prompt = "Enter second number: "))

result <- add_two_numbers(num1, num2)

cat("The sum is:", result, "\\n")
"""

        if lang == "perl":
            return """#!/usr/bin/perl
use strict;
use warnings;

print "Enter first number: ";
my $num1 = <STDIN>;
chomp($num1);

print "Enter second number: ";
my $num2 = <STDIN>;
chomp($num2);

my $sum = $num1 + $num2;

print "The sum is: $sum\\n";
"""

        if lang == "c++":
            return """#include <iostream>
using namespace std;

int main() {
    double num1, num2;

    cout << "Enter first number: ";
    cin >> num1;

    cout << "Enter second number: ";
    cin >> num2;

    cout << "The sum is: " << num1 + num2 << endl;

    return 0;
}
"""

        if lang == "java":
            return """import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter first number: ");
        double num1 = scanner.nextDouble();

        System.out.print("Enter second number: ");
        double num2 = scanner.nextDouble();

        System.out.println("The sum is: " + (num1 + num2));

        scanner.close();
    }
}
"""

        if lang == "html" or lang == "javascript":
            return """<!DOCTYPE html>
<html>
<head>
    <title>Add Two Numbers</title>
</head>
<body>
    <h2>Add Two Numbers</h2>

    <input id="num1" type="number" placeholder="First number">
    <input id="num2" type="number" placeholder="Second number">
    <button onclick="addNumbers()">Add</button>

    <p id="result"></p>

    <script>
        function addNumbers() {
            const num1 = parseFloat(document.getElementById("num1").value);
            const num2 = parseFloat(document.getElementById("num2").value);
            const sum = num1 + num2;

            document.getElementById("result").innerText = "The sum is: " + sum;
        }
    </script>
</body>
</html>
"""

        return """# Python program to add two numbers

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    result = num1 + num2

    print("The sum is:", result)

except ValueError:
    print("Error: Please enter valid numbers.")
"""

    return """# Fallback starter code

print("Qwen model could not generate code on this server.")
print("Try a shorter prompt, reduce output length, or use a Render instance with more RAM.")
"""


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
    if "javascript" in text or "html" in text or "css" in text:
        return "html"
    if "sql" in text:
        return "sql"
    if "php" in text:
        return "php"
    if "bash" in text or "shell" in text:
        return "sh"
    if "typescript" in text:
        return "ts"
    if "go" in text:
        return "go"
    if "rust" in text:
        return "rs"

    return "txt"


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter a coding command.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    code = ""

    try:
        if not QWEN_READY:
            status.warning("Qwen dependencies are not available. Using fallback generator.")
            code = fallback_code(command, language)

            if show_debug:
                st.exception(IMPORT_ERROR)

        else:
            status.info("Step 1/3: Loading Qwen model...")
            progress.progress(25)

            tokenizer, model = load_qwen_model()

            status.info("Step 2/3: Preparing prompt...")
            progress.progress(50)

            status.info("Step 3/3: Generating code...")
            progress.progress(75)

            code = generate_with_qwen(
                user_command=command,
                selected_language=language,
                token_limit=max_tokens,
                temp=temperature
            )

            progress.progress(100)
            status.success("Code generated successfully.")

    except Exception as error:
        status.error("Qwen failed on this server. Showing fallback code.")
        code = fallback_code(command, language)

        if show_debug:
            with st.expander("Debug Error"):
                st.exception(error)

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
