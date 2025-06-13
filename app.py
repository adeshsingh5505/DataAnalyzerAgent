import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import together
import os

st.set_page_config(page_title="Multi-Agent AI Assistant", layout="wide")

# 🗝️ Load Together API Key
together.api_key = st.secrets["TOGETHER_API_KEY"]

# 📄 File Loader Agent
def load_file(file):
    filetype = file.name.split(".")[-1]
    if filetype in ["csv"]:
        df = pd.read_csv(file)
        return df
    elif filetype in ["xlsx"]:
        df = pd.read_excel(file)
        return df
    elif filetype in ["txt", "md"]:
        return file.read().decode("utf-8")
    elif filetype in ["pdf"]:
        import fitz
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    return "Unsupported file type"

# 🧠 Q&A Agent
def ask_ai(prompt, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    response = together.Complete.create(
        model=model,
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response['output']['choices'][0]['text'].strip()

# 📊 Visualization Agent
def plot_data(df, question):
    st.subheader("📊 Data Visualization")
    if "bar" in question:
        st.bar_chart(df.select_dtypes(include='number'))
    elif "line" in question:
        st.line_chart(df.select_dtypes(include='number'))
    elif "hist" in question or "distribution" in question:
        column = df.select_dtypes(include='number').columns[0]
        fig, ax = plt.subplots()
        df[column].hist(ax=ax)
        st.pyplot(fig)
    else:
        st.info("Try asking to 'plot a bar chart of sales', or 'show distribution'.")

# 🚀 UI
st.title("📚 AI Document Analysis Assistant")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "csv", "xlsx", "txt", "md"])
if uploaded_file:
    st.success("File uploaded. You can now ask questions or visualize data.")

    data = load_file(uploaded_file)

    if isinstance(data, pd.DataFrame):
        st.dataframe(data)

        question = st.text_input("Ask a question about the data or type a graph request")
        if question:
            if any(kw in question.lower() for kw in ["plot", "chart", "graph", "distribution", "histogram"]):
                plot_data(data, question)
            else:
                prompt = f"The following table data:\n{data.head(20).to_string()}\n\nQuestion: {question}"
                answer = ask_ai(prompt)
                st.write("🧠", answer)

    elif isinstance(data, str):
        st.text_area("Document Content", value=data[:1000], height=200)

        question = st.text_input("Ask a question about the document")
        if question:
            prompt = f"{data[:3000]}\n\nQuestion: {question}"
            answer = ask_ai(prompt)
            st.write("🧠", answer)
