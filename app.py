# app.py

import streamlit as st
import pandas as pd
import altair as alt
import pytesseract
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch
from pdfplumber import open as open_pdf
from docx import Document

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Chatbot")

# Initialize session state for history and data
if 'history' not in st.session_state:
    st.session_state.history = []  # list of (role, message)
if 'data' not in st.session_state:
    st.session_state.data = None   # store extracted table or text

# Load Llama-4 model (TogetherAI)
@st.cache_resource
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    model = Llama4ForConditionalGeneration.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                                                           torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model

tokenizer, model = load_llama_model()

# --- FileExtractorAgent ---
uploaded_file = st.file_uploader("Upload a data file (.csv, .xlsx, .pdf, .docx, .txt, image)", 
                                 type=['csv','xlsx','xls','pdf','docx','txt','png','jpg','jpeg'])
if uploaded_file is not None:
    # Determine file type and extract content
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type in ['csv', 'xlsx', 'xls']:
        # Read tabular data
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success(f"Loaded {file_type.upper()} data with shape {df.shape}.")
    elif file_type == 'pdf':
        # Extract text from PDF
        text = ""
        with open_pdf(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        st.session_state.data = text
        st.success("Extracted text from PDF.")
    elif file_type == 'docx' or file_type == 'txt':
        # Read text or docx
        if file_type == 'docx':
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = str(uploaded_file.read(), 'utf-8')
        st.session_state.data = text
        st.success("Loaded text data.")
    elif file_type in ['png','jpg','jpeg']:
        # OCR on image
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.session_state.data = text
        st.success("Extracted text from image via OCR.")
    else:
        st.error("Unsupported file type.")
        
# Display existing conversation history
for role, msg in st.session_state.history:
    if role == 'user':
        st.chat_message("user").write(msg)
    else:
        # If role is 'assistant' and message is dict with chart, handle separately
        st.chat_message("assistant").write(msg)

# --- Chat Input (User Query) ---
user_input = st.chat_input("Ask me about the data!")
if user_input:
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    
    # Determine if it's a visualization request (simple keyword check)
    query = user_input.lower()
    if any(term in query for term in ['plot', 'chart', 'visualize', 'graph', 'histogram', 'bar', 'line']):
        df = st.session_state.data
        if isinstance(df, pd.DataFrame):
            # Example: parse a simple command like "plot column A vs B"
            parts = query.replace('plot','').replace('chart','').replace('graph','').split()
            if len(parts) >= 3:
                x_col, _, y_col = parts[0], parts[1], parts[-1]
                if x_col in df.columns and y_col in df.columns:
                    chart = alt.Chart(df).mark_bar().encode(x=x_col, y=y_col)
                    st.chat_message("assistant").altair_chart(chart, use_container_width=True)
                    st.session_state.history.append(("assistant", f"*Displayed a bar chart of {y_col} vs {x_col}.*"))
                else:
                    st.chat_message("assistant").write("I couldn't find those columns to plot.")
                    st.session_state.history.append(("assistant", "Invalid columns for plotting."))
            else:
                st.chat_message("assistant").write("Please specify what to plot (e.g., 'Plot A vs B').")
                st.session_state.history.append(("assistant", "Need more info to plot."))
        else:
            st.chat_message("assistant").write("No tabular data to plot.")
            st.session_state.history.append(("assistant", "No data for visualization."))
    else:
        # --- QuestionAnsweringAgent ---
        # Prepare chat prompt with previous system messages if needed
        prompt_messages = [{"role": "user", "content": user_input}]
        # If data is text, include it; if DataFrame, maybe attach top rows
        data = st.session_state.data
        if isinstance(data, str) and len(data) > 0:
            prompt_messages.insert(0, {"role": "system", "content": data})
        elif isinstance(data, pd.DataFrame):
            # Embed a small representation of data as text
            prompt_messages.insert(0, {"role": "system", "content": data.head().to_csv(index=False)})
        
        inputs = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(**inputs.to(model.device), max_new_tokens=200)
        answer = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        st.chat_message("assistant").write(answer)
        st.session_state.history.append(("assistant", answer))
