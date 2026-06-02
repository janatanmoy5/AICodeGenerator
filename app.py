import os
import re
import gc
from datetime import datetime

import streamlit as st

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    QWEN_AVAILABLE = True
except Exception:
    QWEN_AVAILABLE = False


st.set_page_config(
    page_title="Free Qwen Multi-Language Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Free Qwen Multi-Language Code Generator")
st.write("Enter any coding command. Qwen will generate copy-paste-ready code.")

MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
)

SUPPORTED_LANGUAGES = [
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

with st.sidebar:
    st.header("Settings")

    language = st.selectbox(
        "Programming language",
        SUPPORTED_LANGUAGES
    )

    max_tokens = st.slider(
        "Output length",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    st.caption("Model: " + MODEL_NAME)

    clear_cache = st.button("Clear model memory")

    if clear_cache:
        st.cache_resource.clear()
        gc.collect()
        st.success("Memory cleared. Reload the app if needed.")


command = st.text_area(
    "Enter your coding command",
    height=230,
    placeholder="""
Examples:
Write Python code to add two numbers
Write R code for PCA analysis
Write SQL query to retrieve patient data
Create HTML CSS JavaScript calculator
Write Java program for student grade calculation
Write Perl script to search a name in a text file
Write C++ code for binary search
"""
)


@st.cache_resource(show_spinner="Loading Qwen model...")
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

    parts.append("You are Qwen Coder, an expert multi-language software engineer.")
    parts.append("Your task is to generate complete, correct, copy-paste-ready code.")
    parts.append("")
    parts.append("Selected language:")
    parts.append(selected_language)
    parts.append("")
    parts.append("User command:")
    parts.append(user_command)
    parts.append("")
    parts.append("Rules:")
    parts.append("1. Generate code in the selected language only.")
    parts.append("2. If Auto Detect is selected, infer the language from the user command.")
    parts.append("3. Support Python, R, Perl, C, C++, Java, JavaScript, TypeScript, HTML, CSS, SQL, PHP, Bash, Go, and Rust.")
    parts.append("4. Write complete runnable code.")
    parts.append("5. Include all imports, libraries, class definitions, or package calls when needed.")
    parts.append("6. Include comments inside the code.")
    parts.append("7. Include input/output handling where useful.")
    parts.append("8. For Java, use public class Main with main method.")
    parts.append("9. For R, use valid R syntax only.")
    parts.append("10. For Perl, include use strict and use warnings.")
    parts.append("11. For HTML/CSS/JavaScript, generate a complete HTML file when possible.")
    parts.append("12. For SQL, write clean SQL with comments.")
    parts.append("13. Do not use paid APIs or API keys.")
    parts.append("14. Do not explain outside the code.")
    parts.append("15. Return only final code.")
    parts.append("")
    parts.append("Final code:")

    return "\n".join(parts)


def clean_output(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    blocks = re.findall(
        r"```(?:\\w+)?\\n(.*?)```",
        text,
        re.DOTALL
    )

    if blocks:
        return "\n\n".join(blocks).strip()

    return text.strip()


def generate_with_qwen(user_command, selected_language, token_limit, temp):
    tokenizer, model = load_qwen_model()

    prompt = build_prompt(
        user_command=user_command,
        selected_language=selected_language
    )

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
            temperature=max(temp, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_output(generated)


def fallback_code(user_command, selected_language):
    text = user_command.lower()
    lang = selected_language.lower()

    if selected_language == "Auto Detect":
        if "r code" in text:
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

        return """# Python program to add two numbers

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    result = num1 + num2

    print("The sum is:", result)

except ValueError:
    print("Error: Please enter valid numbers.")
"""

    return """# Starter code

print("Qwen could not generate code on this server.")
print("Try a shorter command or increase Render memory.")
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

    try:
        if QWEN_AVAILABLE:
            with st.spinner("Qwen is generating your code..."):
                code = generate_with_qwen(
                    user_command=command,
                    selected_language=language,
                    token_limit=max_tokens,
                    temp=temperature
                )
        else:
            code = fallback_code(command, language)

        st.success("Code generated successfully.")

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

    except Exception as error:
        st.error("Qwen could not run on this Render server.")
        st.info("Try a smaller prompt, reduce output length, or use a Render instance with more RAM.")

        fallback = fallback_code(command, language)

        st.subheader("Fallback Code")
        st.code(fallback)

        st.download_button(
            label="⬇️ Download Fallback Code",
            data=fallback,
            file_name="fallback_code.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("Error details"):
            st.exception(error)
