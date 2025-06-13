import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import together
from typing import List, Dict

# -------------------- Together AI Setup --------------------
TOGETHER_API_KEY = st.secrets["0ad00340c320b21832908f303996ced948794a421e54e95da9a4763ffa8792fa"]
TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
together.api_key = TOGETHER_API_KEY

def llama4_infer(messages: List[Dict[str, str]], max_tokens=700) -> str:
    response = together.ChatCompletion.create(
        model=TOGETHER_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

# -------------------- Agents --------------------
class DataAnalysisAgent:
    def __init__(self):
        self.chat_history = []

    def analyze(self, data: pd.DataFrame, question: str) -> str:
        csv_sample = data.head(30).to_csv(index=False)  # Limit for prompt length
        prompt = (
            "You are a data analyst. Given the following data (in CSV format) and the user's question, "
            "provide a concise, professional answer. If you don't know, say so.\n\n"
            f"Data:\n{csv_sample}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        self.chat_history.append({"role": "user", "content": prompt})
        answer = llama4_infer(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

class QAAgent:
    def __init__(self, context: str):
        self.chat_history = [{"role": "system", "content": context}]

    def ask(self, question: str) -> str:
        self.chat_history.append({"role": "user", "content": question})
        answer = llama4_infer(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

class VisualizationAgent:
    def visualize(self, data: pd.DataFrame, chart_type: str, x_col: str, y_col: str):
        plt.figure(figsize=(8, 4))
        if chart_type == "Bar":
            plt.bar(data[x_col], data[y_col])
        elif chart_type == "Line":
            plt.plot(data[x_col], data[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{chart_type} Chart: {y_col} vs {x_col}")
        st.pyplot(plt)
        plt.close()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI Data Analysis & QnA Agent", page_icon="🤖")
st.title("🤖 AI Data Analysis & Q&A Agent (Together AI)")

st.write("Upload a CSV file, ask questions about your data, and generate visualizations. Powered by Together AI Llama-4 Maverick.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head(10))

    # --- Data Analysis Section ---
    st.header("🔎 Data Analysis & Q&A")
    if "qa_agent" not in st.session_state:
        st.session_state.qa_agent = None
    if "data_agent" not in st.session_state:
        st.session_state.data_agent = DataAnalysisAgent()

    user_question = st.text_input("Ask a question about your data (follow-ups supported!)")
    if st.button("Analyze / Ask"):
        if user_question.strip():
            # Data analysis agent (with chat history for follow-ups)
            answer = st.session_state.data_agent.analyze(data, user_question)
            st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

    # --- Visualization Section ---
    st.header("📊 Data Visualization")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line"])
    x_col = st.selectbox("X Axis", data.columns)
    y_col = st.selectbox("Y Axis", data.columns)
    if st.button("Create Chart"):
        try:
            VisualizationAgent().visualize(data, chart_type, x_col, y_col)
        except Exception as e:
            st.error(f"Error creating chart: {e}")

    # --- General Q&A Section (about the data, not the CSV) ---
    st.header("💬 General Data Q&A (with context)")
    if st.session_state.qa_agent is None:
        # Use a summary of the data as context
        context = f"Columns: {list(data.columns)}\nFirst rows:\n{data.head(5).to_csv(index=False)}"
        st.session_state.qa_agent = QAAgent(context)
    user_q = st.text_input("Ask a general question (context-aware):", key="genq")
    if st.button("Ask General Q"):
        if user_q.strip():
            answer = st.session_state.qa_agent.ask(user_q)
            st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.caption("Built with Together AI Llama-4 Maverick | [Get your free API key](https://www.together.ai/)")

