import streamlit as st
from datetime import datetime
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    QWEN_AVAILABLE = True
except Exception:
    QWEN_AVAILABLE = False


st.set_page_config(
    page_title="Free Multi-Language AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Free Multi-Language AI Code Generator")
st.write(
    "Enter any coding command. The app generates code logic, syntax, and ready-to-copy output."
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

language = st.selectbox(
    "Choose programming language",
    [
        "Auto Detect",
        "Python",
        "R",
        "Perl",
        "C++",
        "HTML",
        "CSS",
        "HTML + CSS + JavaScript",
        "Java",
        "JavaScript",
        "SQL",
        "PHP",
        "Bash"
    ]
)

command = st.text_area(
    "Enter your command",
    height=220,
    placeholder="Example: Write R code to add two numbers"
)

max_tokens = st.slider(
    "Output length",
    min_value=256,
    max_value=2048,
    value=1024,
    step=256
)


@st.cache_resource
def load_qwen_model():
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
    parts = []

    parts.append("You are an expert multi-language software engineer.")
    parts.append("You generate correct, clean, complete, copy-paste-ready code.")
    parts.append("")
    parts.append("Selected language:")
    parts.append(selected_language)
    parts.append("")
    parts.append("User command:")
    parts.append(user_command)
    parts.append("")
    parts.append("Important rules:")
    parts.append("1. Generate code in the selected language only.")
    parts.append("2. If Auto Detect is selected, infer language from the command.")
    parts.append("3. Write complete executable code.")
    parts.append("4. Include imports, libraries, or package calls if needed.")
    parts.append("5. Include comments inside the code.")
    parts.append("6. Include input and output handling where useful.")
    parts.append("7. Do not use API keys.")
    parts.append("8. Do not explain outside the code.")
    parts.append("9. Return only final code.")
    parts.append("10. For Java, include public class Main with main method.")
    parts.append("11. For HTML/CSS/JavaScript, include a full HTML document.")
    parts.append("12. For R, use valid R syntax, not Python.")
    parts.append("13. For Perl, include strict and warnings.")
    parts.append("14. For SQL, include clean query with comments.")
    parts.append("")
    parts.append("Final code:")

    return "\n".join(parts)


def clean_generated_text(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1]

    blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_with_qwen(user_command, selected_language, token_limit):
    tokenizer, model = load_qwen_model()

    prompt = build_prompt(user_command, selected_language)

    messages = [
        {
            "role": "system",
            "content": "You are Qwen Coder. Return only final runnable code."
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
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_generated_text(result)


def fallback_code(user_command, selected_language):
    text = user_command.lower()
    lang = selected_language.lower()

    if selected_language == "Auto Detect":
        if " r " in " " + text + " " or "r code" in text:
            lang = "r"
        elif "perl" in text:
            lang = "perl"
        elif "c++" in text or "cpp" in text:
            lang = "c++"
        elif "html" in text:
            lang = "html"
        elif "css" in text:
            lang = "css"
        elif "java " in text:
            lang = "java"
        elif "javascript" in text or " js " in " " + text + " ":
            lang = "javascript"
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
            return '''# R program to add two numbers

add_two_numbers <- function(a, b) {
  return(a + b)
}

num1 <- as.numeric(readline(prompt = "Enter first number: "))
num2 <- as.numeric(readline(prompt = "Enter second number: "))

result <- add_two_numbers(num1, num2)

cat("The sum is:", result, "\\n")
'''

        if lang == "perl":
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

        if lang == "c++":
            return '''#include <iostream>
using namespace std;

int main() {
    double num1, num2, sum;

    cout << "Enter first number: ";
    cin >> num1;

    cout << "Enter second number: ";
    cin >> num2;

    sum = num1 + num2;

    cout << "The sum is: " << sum << endl;

    return 0;
}
'''

        if lang == "java":
            return '''import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter first number: ");
        double num1 = scanner.nextDouble();

        System.out.print("Enter second number: ");
        double num2 = scanner.nextDouble();

        double sum = num1 + num2;

        System.out.println("The sum is: " + sum);

        scanner.close();
    }
}
'''

        if lang == "javascript":
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
            let num1 = parseFloat(document.getElementById("num1").value);
            let num2 = parseFloat(document.getElementById("num2").value);

            let sum = num1 + num2;

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

    if "html" in lang:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Generated Web Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            padding: 40px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Hello World</h1>
        <p>This page was generated by the code generator.</p>
    </div>
</body>
</html>
'''

    if "css" in lang:
        return '''/* Clean responsive CSS template */

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background-color: #f4f6f8;
}

.container {
    max-width: 900px;
    margin: auto;
    padding: 30px;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
'''

    if "sql" in lang:
        return '''-- SQL query template

SELECT
    *
FROM
    table_name
WHERE
    condition_column = 'value';
'''

    if "php" in lang:
        return '''<?php
// PHP starter program

function greet_user($name) {
    return "Hello, " . $name . "!";
}

echo greet_user("Tanmoy");
?>
'''

    if "bash" in lang:
        return '''#!/bin/bash

# Bash starter script

echo "Enter your name:"
read name

echo "Hello, $name"
'''

    return '''# Python starter program

def main():
    print("Code generator is ready.")
    print("Please enter a more specific programming command.")

if __name__ == "__main__":
    main()
'''


def get_extension(selected_language, user_command):
    text = (selected_language + " " + user_command).lower()

    if "python" in text or "streamlit" in text:
        return "py"
    if " r " in " " + text + " " or selected_language == "R":
        return "R"
    if "perl" in text:
        return "pl"
    if "c++" in text or "cpp" in text:
        return "cpp"
    if "html" in text or "css" in text or "javascript" in text:
        return "html"
    if "java" in text:
        return "java"
    if "sql" in text:
        return "sql"
    if "php" in text:
        return "php"
    if "bash" in text:
        return "sh"

    return "txt"


if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter a coding command.")
        st.stop()

    code = ""

    if QWEN_AVAILABLE:
        try:
            with st.spinner("Qwen is generating code..."):
                code = generate_with_qwen(
                    command,
                    language,
                    max_tokens
                )
        except Exception:
            code = fallback_code(command, language)
    else:
        code = fallback_code(command, language)

    st.success("Code generated successfully.")

    st.subheader("Generated Code")
    st.code(code)

    extension = get_extension(language, command)

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
