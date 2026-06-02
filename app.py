import os
import re
import gc
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")

import streamlit as st

st.set_page_config(
    page_title="Qwen Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen Code Generator")
st.write("Write any command. The app generates code. If Qwen is too slow, fallback code appears immediately.")

MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

language = st.selectbox(
    "Programming Language",
    [
        "Auto Detect", "Python", "R", "Perl", "C", "C++", "Java",
        "JavaScript", "HTML", "CSS", "SQL", "PHP", "Bash"
    ]
)

command = st.text_area(
    "Enter your coding command",
    height=200,
    placeholder="Example: Write R code to add two numbers"
)

max_tokens = st.slider(
    "Output Length",
    min_value=64,
    max_value=512,
    value=256,
    step=64
)

use_qwen = st.checkbox(
    "Use Qwen model",
    value=True
)

show_debug = st.checkbox(
    "Show debug info",
    value=False
)


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

    return "Python"


def fallback_code(user_command, selected_language):
    text = user_command.lower()
    lang = detect_language(user_command, selected_language)

    if "add" in text and "number" in text:
        if lang == "R":
            return """# R program to add two numbers

add_two_numbers <- function(a, b) {
  return(a + b)
}

num1 <- as.numeric(readline(prompt = "Enter first number: "))
num2 <- as.numeric(readline(prompt = "Enter second number: "))

result <- add_two_numbers(num1, num2)

cat("The sum is:", result, "\\n")
"""

        if lang == "Perl":
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

        if lang == "C++":
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

        if lang == "C":
            return """#include <stdio.h>

int main() {
    double num1, num2;

    printf("Enter first number: ");
    scanf("%lf", &num1);

    printf("Enter second number: ");
    scanf("%lf", &num2);

    printf("The sum is: %.2f\\n", num1 + num2);

    return 0;
}
"""

        if lang == "Java":
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

        if lang == "JavaScript":
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
            document.getElementById("result").innerText = "The sum is: " + (num1 + num2);
        }
    </script>
</body>
</html>
"""

        return """# Python program to add two numbers

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    print("The sum is:", num1 + num2)

except ValueError:
    print("Error: Please enter valid numbers.")
"""

    return f"""# {lang} starter code

# Request:
# {user_command}

print("This request needs Qwen model generation.")
print("If Qwen is slow on Render, reduce output length or upgrade memory.")
"""


def build_prompt(user_command, selected_language):
    return f"""
You are Qwen Coder.

Generate complete runnable code only.

Language:
{selected_language}

User command:
{user_command}

Rules:
- Return only code.
- No explanation outside code.
- Code must be copy-paste ready.
- If Auto Detect, infer the language.
- Support Python, R, Perl, C, C++, Java, JavaScript, HTML, CSS, SQL, PHP, Bash.
- For Java use public class Main.
- For Perl use strict and warnings.
- For HTML/JavaScript create full HTML if useful.

Final code:
"""


def clean_output(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    blocks = re.findall(r"```(?:\\w+)?\\n(.*?)```", text, re.DOTALL)

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_with_qwen(user_command, selected_language, token_limit):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Return only code."
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
        return_tensors="pt",
        truncation=True,
        max_length=768
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

    del tokenizer
    del model
    gc.collect()

    return clean_output(result)


def get_extension(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    mapping = {
        "Python": "py",
        "R": "R",
        "Perl": "pl",
        "C": "c",
        "C++": "cpp",
        "Java": "java",
        "JavaScript": "html",
        "HTML": "html",
        "CSS": "css",
        "SQL": "sql",
        "PHP": "php",
        "Bash": "sh"
    }

    return mapping.get(lang, "txt")


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter a command.")
        st.stop()

    immediate_code = fallback_code(command, language)

    st.subheader("Immediate Output")
    st.code(immediate_code)

    code = immediate_code

    if use_qwen:
        try:
            with st.spinner("Trying Qwen generation. If server memory is low, fallback will remain."):
                qwen_code = generate_with_qwen(
                    command,
                    language,
                    max_tokens
                )

            if qwen_code and len(qwen_code.strip()) > 20:
                code = qwen_code
                st.subheader("Qwen Generated Code")
                st.code(code)
            else:
                st.warning("Qwen returned empty output. Fallback code shown above.")

        except Exception as error:
            st.warning("Qwen failed or timed out on this server. Fallback code shown above.")

            if show_debug:
                with st.expander("Debug details"):
                    st.exception(error)

    file_name = (
        "generated_code_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + get_extension(command, language)
    )

    st.download_button(
        label="⬇️ Download Final Code",
        data=code,
        file_name=file_name,
        mime="text/plain",
        use_container_width=True
    )
