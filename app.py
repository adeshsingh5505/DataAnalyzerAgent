import streamlit as st
import pandas as pd
import altair as alt
import pytesseract
from PIL import Image
import together
from io import BytesIO
from docx import Document
import fitz  # PyMuPDF

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Assistant (Together AI - Llama 4)")

# Initialize Together API
together.api_key = st.secrets["TOGETHER_API_KEY"]

# State initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "data" not in st.session_state:
    st.session_state.data = None

# --- File Extraction Agent ---
uploaded_file = st.file_uploader("Upload your document", type=["csv", "xlsx", "xls", "pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file:
    filetype = uploaded_file.name.split(".")[-1].lower()
    try:
        if filetype == "csv":
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.dataframe(df)
        elif filetype in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.dataframe(df)
        elif filetype == "pdf":
            text = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            st.session_state.data = text
            st.text_area("Extracted Text", text[:2000])
        elif filetype == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            st.session_state.data = text
            st.text_area("Extracted Text", text[:2000])
        elif filetype == "txt":
            text = uploaded_file.read().decode("utf-8")
            st.session_state.data = text
            st.text_area("Text File Content", text[:2000])
        elif filetype in ["jpg", "jpeg", "png"]:
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            st.session_state.data = text
            st.image(img, caption="Uploaded Image")
            st.text_area("OCR Text", text[:2000])
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Failed to process file: {e}")

# --- Ask Together AI ---
def ask_together(prompt):
    try:
        response = together.Complete.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )
        return response['output']['choices'][0]['text'].strip()
    except Exception as e:
        return f"❌ Error contacting Together AI: {str(e)}"

# --- Chat UI ---
user_query = st.chat_input("Ask a question about the file or request a plot...")
if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.history.append(("user", user_query))

    if "plot" in user_query or "chart" in user_query:
        if isinstance(st.session_state.data, pd.DataFrame):
            df = st.session_state.data
            try:
                chart = alt.Chart(df).mark_bar().encode(x=df.columns[0], y=df.columns[1])
                st.chat_message("assistant").altair_chart(chart, use_container_width=True)
                st.session_state.history.append(("assistant", f"📊 Plotted {df.columns[0]} vs {df.columns[1]}"))
            except:
                st.chat_message("assistant").write("Couldn't generate plot from the data.")
        else:
            st.chat_message("assistant").write("No structured data to plot.")
    else:
        content = st.session_state.data
        context = content[:2000] if isinstance(content, str) else content.head(5).to_string() if isinstance(content, pd.DataFrame) else ""
        full_prompt = f"{context}\n\nUser Question: {user_query}"
        answer = ask_together(full_prompt)
        st.chat_message("assistant").write(answer)
        st.session_state.history.append(("assistant", answer))
