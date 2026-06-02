import os
import re
import requests
from datetime import datetime

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st

st.set_page_config(
    page_title="Qwen AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen AI Code Generator")
st.write("Enter any coding instruction. The app gives immediate code and can improve it using Qwen/Hugging Face.")

MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

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
        min_value=64,
        max_value=2048,
        value=768,
        step=64
    )

    use_qwen = st.checkbox("Use Qwen/Hugging Face generation", value=True)

    hf_token_input = st.text_input(
        "Hugging Face Token Optional",
        type="password"
    )

    show_debug = st.checkbox("Show debug info", value=True)

command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: write R code to add two numbers"
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
    if "typescript" in text:
        return "TypeScript"
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
    if "go " in f" {text} ":
        return "Go"
    if "rust" in text:
        return "Rust"

    return "Python"

def get_extension(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    mapping = {
        "Python": "py",
        "R": "R",
        "Perl": "pl",
        "C": "c",
        "C++": "cpp",
        "Java": "java",
        "JavaScript": "js",
        "TypeScript": "ts",
        "HTML": "html",
        "CSS": "css",
        "HTML + CSS + JavaScript": "html",
        "SQL": "sql",
        "PHP": "php",
        "Bash": "sh",
        "Go": "go",
        "Rust": "rs"
    }

    return mapping.get(lang, "txt")

def fallback_code(user_command, selected_language):
    text = user_command.lower()
    lang = detect_language(user_command, selected_language)

    if "add" in text and "number" in text:

        if lang == "R":
            return '''# R program to add two numbers

add_two_numbers <- function(a, b) {
  return(a + b)
}

num1 <- as.numeric(readline(prompt = "Enter first number: "))
num2 <- as.numeric(readline(prompt = "Enter second number: "))

result <- add_two_numbers(num1, num2)

cat("The sum is:", result, "\\n")
'''

        if lang == "Perl":
            return '''#!/usr/bin/perl
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
'''

        if lang == "C":
            return '''#include <stdio.h>

int main() {
    double num1, num2;

    printf("Enter first number: ");
    scanf("%lf", &num1);

    printf("Enter second number: ");
    scanf("%lf", &num2);

    printf("The sum is: %.2f\\n", num1 + num2);

    return 0;
}
'''

        if lang == "C++":
            return '''#include <iostream>
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
'''

        if lang == "Java":
            return '''import java.util.Scanner;

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
'''

        if lang in ["HTML", "JavaScript", "HTML + CSS + JavaScript"]:
            return '''<!DOCTYPE html>
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
'''

        return '''# Python program to add two numbers

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    result = num1 + num2

    print("The sum is:", result)

except ValueError:
    print("Error: Please enter valid numbers.")
'''

    if "name" in text and "tanmoy" in text:

        if lang == "HTML":
            return '''<!DOCTYPE html>
<html>
<head>
    <title>Print Name</title>
</head>
<body>
    <h1>My name is Tanmoy</h1>
</body>
</html>
'''

        if lang == "R":
            return '''cat("My name is Tanmoy\\n")
'''

        if lang == "Perl":
            return '''#!/usr/bin/perl
use strict;
use warnings;

print "My name is Tanmoy\\n";
'''

        if lang == "Java":
            return '''public class Main {
    public static void main(String[] args) {
        System.out.println("My name is Tanmoy");
    }
}
'''

        if lang == "C++":
            return '''#include <iostream>
using namespace std;

int main() {
    cout << "My name is Tanmoy" << endl;
    return 0;
}
'''

        if lang == "C":
            return '''#include <stdio.h>

int main() {
    printf("My name is Tanmoy\\n");
    return 0;
}
'''

        if lang == "SQL":
            return '''SELECT 'My name is Tanmoy' AS message;
'''

        return '''print("My name is Tanmoy")
'''

    if lang == "SQL":
        return '''-- SQL query template

SELECT
    *
FROM
    table_name
WHERE
    condition_column = 'value';
'''

    if lang == "HTML":
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Generated Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This page was generated from your instruction.</p>
</body>
</html>
'''

    if lang == "CSS":
        return '''body {
    font-family: Arial, sans-serif;
    background-color: #f5f7fa;
    color: #222;
}

.container {
    max-width: 900px;
    margin: auto;
    padding: 30px;
}
'''

    if lang == "PHP":
        return '''<?php
$message = "Hello from PHP";
echo $message;
?>
'''

    return f'''# Starter code

# Language: {lang}
# User request:
# {user_command}

print("Qwen can generate the full custom code for this request.")
'''

def build_prompt(user_command, selected_language):
    lang = detect_language(user_command, selected_language)

    return f'''You are Qwen Coder, an expert programming assistant.

Generate clean, complete, runnable code.

Language:
{lang}

User instruction:
{user_command}

Rules:
1. Return only code.
2. No explanation outside code.
3. Code must be copy-paste ready.
4. Use the requested programming language only.
5. Include imports or libraries if needed.
6. For Java, use public class Main.
7. For R, use valid R syntax.
8. For Perl, include use strict and use warnings.
9. For HTML/CSS/JavaScript, return complete HTML if useful.
'''

def clean_output(text):
    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    return text.strip()

def call_qwen_hf(user_command, selected_language, token_limit, token):
    headers = {
        "Content-Type": "application/json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    prompt = build_prompt(user_command, selected_language)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": token_limit,
            "temperature": 0.2,
            "return_full_text": False
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    data = response.json()

    if isinstance(data, list) and len(data) > 0:
        if "generated_text" in data[0]:
            return clean_output(data[0]["generated_text"])

    if isinstance(data, dict) and "generated_text" in data:
        return clean_output(data["generated_text"])

    return clean_output(str(data))

if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter your coding instruction.")
        st.stop()

    instant_code = fallback_code(command, language)

    st.subheader("Immediate Code Output")
    st.code(instant_code)

    final_code = instant_code

    if use_qwen:
        try:
            with st.spinner("Generating improved Qwen code..."):
                final_code = call_qwen_hf(
                    user_command=command,
                    selected_language=language,
                    token_limit=max_tokens,
                    token=hf_token_input
                )

            st.success("Qwen code generated successfully.")

            st.subheader("Qwen Generated Code")
            st.code(final_code)

        except Exception as error:
            st.warning("Qwen could not generate code. Immediate code is still available.")

            if show_debug:
                with st.expander("Debug information"):
                    st.exception(error)

    ext = get_extension(command, language)

    file_name = (
        "generated_code_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + ext
    )

    st.download_button(
        label="⬇️ Download Code",
        data=final_code,
        file_name=file_name,
        mime="text/plain",
        use_container_width=True
    )
