import os
import re
import multiprocessing as mp
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")

import streamlit as st

st.set_page_config(page_title="Qwen Code Generator", page_icon="🤖", layout="wide")

MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

st.title("🤖 Qwen Code Generator")
st.write("Write any command. Fallback appears immediately. Qwen output appears only if it finishes fast.")

language = st.selectbox(
    "Programming Language",
    [
        "Auto Detect", "Python", "R", "Perl", "C", "C++", "Java",
        "JavaScript", "HTML", "CSS", "SQL", "PHP", "Bash"
    ]
)

command = st.text_area(
    "Enter your coding instruction",
    height=220,
    placeholder="Example: Write Python code to add two numbers"
)

max_tokens = st.slider("Output Length", 64, 384, 192, 64)

qwen_timeout = st.slider(
    "Qwen Timeout Seconds",
    min_value=10,
    max_value=90,
    value=30,
    step=10
)

use_qwen = st.checkbox("Try Qwen model", value=True)


def detect_language(cmd, selected):
    text = cmd.lower()

    if selected != "Auto Detect":
        return selected

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


def fallback_code(cmd, selected):
    text = cmd.lower()
    lang = detect_language(cmd, selected)

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

        if lang == "C++":
            return '''#include <iostream>
using namespace std;

int main() {
    double a, b;

    cout << "Enter first number: ";
    cin >> a;

    cout << "Enter second number: ";
    cin >> b;

    cout << "Sum: " << a + b << endl;

    return 0;
}
'''

        if lang == "JavaScript" or lang == "HTML":
            return '''<!DOCTYPE html>
<html>
<head>
    <title>Add Two Numbers</title>
</head>
<body>
    <h2>Add Two Numbers</h2>

    <input id="a" type="number" placeholder="First number">
    <input id="b" type="number" placeholder="Second number">
    <button onclick="addNumbers()">Add</button>

    <p id="result"></p>

    <script>
        function addNumbers() {
            const a = parseFloat(document.getElementById("a").value);
            const b = parseFloat(document.getElementById("b").value);

            document.getElementById("result").innerText = "Sum: " + (a + b);
        }
    </script>
</body>
</html>
'''

        return '''# Python program to add two numbers

try:
    a = float(input("Enter first number: "))
    b = float(input("Enter second number: "))

    print("Sum:", a + b)

except ValueError:
    print("Error: Please enter valid numbers.")
'''

    return f'''# Starter code generated immediately

# Language: {lang}
# User request:
# {cmd}

print("This request needs Qwen AI generation.")
print("Qwen did not finish quickly, so this fallback output is shown.")
'''


def clean_output(text):
    if "Final code:" in text:
        text = text.split("Final code:")[-1].strip()

    blocks = re.findall(r"```(?:\\w+)?\\n(.*?)```", text, re.DOTALL)

    if blocks:
        return "\\n\\n".join(blocks).strip()

    return text.strip()


def build_prompt(cmd, selected):
    return f'''
You are Qwen Coder.

Generate only complete runnable code.

Language:
{selected}

User instruction:
{cmd}

Rules:
- Return only code.
- No explanation outside code.
- Code must be copy-paste ready.
- If Auto Detect, infer language.
- Support Python, R, Perl, C, C++, Java, JavaScript, HTML, CSS, SQL, PHP, Bash.
- For Java use public class Main.
- For Perl use strict and warnings.
- For HTML/JavaScript create full HTML if useful.

Final code:
'''


def qwen_worker(cmd, selected, token_limit, queue):
    try:
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

        prompt = build_prompt(cmd, selected)

        messages = [
            {"role": "system", "content": "You are Qwen Coder. Return only code."},
            {"role": "user", "content": prompt}
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
            max_length=512
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        queue.put({
            "ok": True,
            "code": clean_output(result)
        })

    except Exception as e:
        queue.put({
            "ok": False,
            "error": str(e)
        })


def run_qwen_with_timeout(cmd, selected, token_limit, timeout_seconds):
    queue = mp.Queue()

    process = mp.Process(
        target=qwen_worker,
        args=(cmd, selected, token_limit, queue)
    )

    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, "Qwen timeout. Fallback output was used."

    if not queue.empty():
        result = queue.get()

        if result.get("ok"):
            return result.get("code"), None

        return None, result.get("error")

    return None, "Qwen returned no output."


def get_extension(cmd, selected):
    lang = detect_language(cmd, selected)

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


if "final_code" not in st.session_state:
    st.session_state.final_code = ""

if st.button("🚀 Generate Code", use_container_width=True):
    if not command.strip():
        st.warning("Please enter a coding instruction.")
        st.stop()

    immediate = fallback_code(command, language)
    st.session_state.final_code = immediate

    st.subheader("Immediate Output")
    st.code(immediate)

    if use_qwen:
        with st.spinner("Trying Qwen generation with timeout protection..."):
            qwen_code, error = run_qwen_with_timeout(
                command,
                language,
                max_tokens,
                qwen_timeout
            )

        if qwen_code and len(qwen_code.strip()) > 20:
            st.session_state.final_code = qwen_code
            st.success("Qwen generated code successfully.")
        else:
            st.warning(error or "Qwen did not generate output. Fallback output used.")

if st.session_state.final_code:
    st.subheader("Final Downloadable Code")
    st.code(st.session_state.final_code)

    filename = (
        "generated_code_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "."
        + get_extension(command, language)
    )

    st.download_button(
        label="⬇️ Download Code",
        data=st.session_state.final_code,
        file_name=filename,
        mime="text/plain",
        use_container_width=True
    )
