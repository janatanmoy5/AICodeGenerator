import os
import re
import gc
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")

import streamlit as st

st.set_page_config(
    page_title="Qwen AI Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Qwen AI Code Generator")
st.write(
    "Write any command. The app shows immediate code first, then Qwen-generated code if the model finishes successfully."
)

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
        value=True
    )

    if st.button("Clear memory"):
        st.cache_resource.clear()
        gc.collect()
        st.success("Memory cleared.")


command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: write html code to print my name is Tanmoy"
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

    if "name" in text and "tanmoy" in text:
        if lang == "HTML":
            return """<!DOCTYPE html>
<html>
<head>
    <title>Print Name</title>
</head>
<body>
    <h1>My name is Tanmoy</h1>
</body>
</html>
"""

        if lang == "CSS":
            return """body {
    font-family: Arial, sans-serif;
    background-color: #f5f7fa;
    color: #222;
}

h1 {
    color: #0057b8;
    text-align: center;
}
"""

        if lang == "JavaScript":
            return """<!DOCTYPE html>
<html>
<head>
    <title>Print Name</title>
</head>
<body>
    <h1 id="output"></h1>

    <script>
        document.getElementById("output").innerText = "My name is Tanmoy";
    </script>
</body>
</html>
"""

        if lang == "PHP":
            return """<?php
$name = "Tanmoy";
echo "My name is " . $name;
?>
"""

        if lang == "R":
            return """cat("My name is Tanmoy\n")
"""

        if lang == "Perl":
            return """#!/usr/bin/perl
use strict;
use warnings;

print "My name is Tanmoy\n";
"""

        if lang == "C":
            return """#include <stdio.h>

int main() {
    printf("My name is Tanmoy\n");
    return 0;
}
"""

        if lang == "C++":
            return """#include <iostream>
using namespace std;

int main() {
    cout << "My name is Tanmoy" << endl;
    return 0;
}
"""

        if lang == "Java":
            return """public class Main {
    public static void main(String[] args) {
        System.out.println("My name is Tanmoy");
    }
}
"""

        if lang == "SQL":
            return """SELECT 'My name is Tanmoy' AS message;
"""

        return """print("My name is Tanmoy")
"""

    if "add" in text and "number" in text:
        if lang == "R":
            return """# R program to add two numbers

add_two_numbers <- function(a, b) {
  return(a + b)
}

num1 <- as.numeric(readline(prompt = "Enter first number: "))
num2 <- as.numeric(readline(prompt = "Enter second number: "))

result <- add_two_numbers(num1, num2)

cat("The sum is:", result, "\n")
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

print "The sum is: $sum\n";
"""

        if lang == "C":
            return """#include <stdio.h>

int main() {
    double num1, num2;

    printf("Enter first number: ");
    scanf("%lf", &num1);

    printf("Enter second number: ");
    scanf("%lf", &num2);

    printf("The sum is: %.2f\n", num1 + num2);

    return 0;
}
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

        if lang in ["JavaScript", "HTML", "HTML + CSS + JavaScript"]:
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

    if lang == "HTML":
        return """<!DOCTYPE html>
<html>
<head>
    <title>Generated HTML Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This page was generated from your instruction.</p>
</body>
</html>
"""

    if lang == "PHP":
        return """<?php
$message = "Hello from PHP";
echo $message;
?>
"""

    if lang == "SQL":
        return """-- SQL query template

SELECT
    *
FROM
    table_name
WHERE
    condition_column = 'value';
"""

    return f"""# Starter code

# Language: {lang}
# User request:
# {user_command}

print("Qwen is required for this custom request.")
print("If Qwen is slow, reduce output length or use a stronger Render instance.")
"""


def build_prompt(user_command, selected_language):
    return f"""
You are Qwen Coder, a fast expert programming assistant.

Selected language:
{selected_language}

User instruction:
{user_command}

Rules:
1. Generate only code.
2. Do not explain outside the code.
3. Code must be copy-paste ready.
4. If Auto Detect is selected, infer language from instruction.
5. Support Python, R, Perl, C, C++, Java, JavaScript, TypeScript, HTML, CSS, SQL, PHP, Bash, Go, and Rust.
6. For Java, use public class Main.
7. For Perl, include use strict and use warnings.
8. For HTML/JavaScript/CSS, generate a full runnable HTML file when useful.
9. For SQL, include clean SQL comments.
10. Keep code complete but concise.
11. Return only the final code.

Final code:
"""


def clean_output(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    blocks = re.findall(
        r"
http://googleusercontent.com/immersive_entry_chip/0

### 📋 Important Deployment Requirement
Make sure your `requirements.txt` file explicitly includes `accelerate` so that the `device_map="auto"` argument works properly in low-memory environments:
```text
streamlit
torch
transformers
accelerate
