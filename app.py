import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import together
import base64
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import docx

together.api_key = st.secrets["TOGETHER_API_KEY"]

def ask_together(prompt: str, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    response = together.Complete.create(
        model=model,
        prompt=prompt,
        max_tokens=512,
        temperature=0.7
    )
    return response['output']['choices'][0]['text'].strip()

def extract_text(file) -> str:
    ext = file.name.split('.')[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "csv":
        df = pd.read_csv(file)
        return df.head().to_string()
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(file)
        return df.head().to_string()
    elif ext == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif ext == "docx":
        document = docx.Document(file)
        return "\n".join([p.text for p in document.paragraphs])
    elif ext in ["png", "jpg", "jpeg"]:
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    else:
        return "Unsupported file format."

def generate_plot_from_csv(file):
    df = pd.read_csv(file)
    st.subheader("📊 Histogram of All Numeric Columns")
    df.hist(figsize=(10, 5))
    st.pyplot(plt)

st.set_page_config(page_title="📁 Smart File Analyzer Agent", layout="wide")
st.title("📁 Smart Document Agent with Together AI")

uploaded_file = st.file_uploader(
    "Upload a file (.txt, .docx, .csv, .xlsx, .pdf, .jpg, .png)", 
    type=["txt", "docx", "csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()

    st.subheader("📄 Preview Extracted Text")
    text = extract_text(uploaded_file)
    st.text(text[:3000])

    st.subheader("❓ Ask a Question about This Document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("Analyzing..."):
            full_prompt = f"Document Content:\n{text}\n\nQuestion: {user_question}\nAnswer:"
            response = ask_together(full_prompt)
        st.success("✅ Answer:")
        st.markdown(response)

    if ext == "csv":
        if st.button("📊 Generate Visualization"):
            generate_plot_from_csv(uploaded_file)
