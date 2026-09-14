import streamlit as st
from src.rag import ask
from src.vector_store import build_vector_store

st.set_page_config(page_title="IntelliBot", page_icon="🤖", layout="centered")


st.markdown("""
<style>

    .block-container {
        padding-top: 2.5rem;
        max-width: 780px;
    }


    h1 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }


    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.25rem 0.5rem;
    }


    section[data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }


    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }


    [data-testid="stChatInput"] textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Darshan's IntelliBot")
st.caption("Ask questions about your uploaded documents.")

if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.subheader("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Add PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Add to knowledge base", use_container_width=True):
        for file in uploaded_files:
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        with st.spinner("Embedding documents..."):
            build_vector_store()
        st.success(f"Added {len(uploaded_files)} file(s).")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


if not st.session_state.history:
    st.info("No messages yet — ask something about your documents below.")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])


query = st.chat_input("Ask something about your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(query, st.session_state.history)
        st.write(answer)

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})