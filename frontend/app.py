import os
import uuid
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Production RAG Chatbot",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("📄 Production RAG Chatbot")

st.write("Upload a PDF and ask questions from the document.")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# =========================
# EMBEDDING MODEL
# =========================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# PROCESS PDF
# =========================

if uploaded_file is not None:

    # SAVE PDF
    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # =========================
    # LOAD PDF
    # =========================

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    st.write(f"Loaded {len(documents)} pages.")

    # =========================
    # SPLIT DOCUMENTS
    # =========================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    st.write(f"Created {len(docs)} chunks.")

    # =========================
    # CREATE UNIQUE DB PATH
    # =========================

    db_path = f"temp_db_{uuid.uuid4()}"

    # =========================
    # CREATE VECTOR STORE
    # =========================

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=db_path
    )

    # =========================
    # CREATE RETRIEVER
    # =========================

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # =========================
    # LOAD LOCAL LLM
    # =========================

    llm = OllamaLLM(
        model="tinyllama"
    )

    # =========================
    # QUESTION INPUT
    # =========================

    query = st.text_input(
        "Ask a question about the PDF:"
    )

    # =========================
    # ASK BUTTON
    # =========================

    if st.button("Ask"):

        if query.strip() == "":
            st.warning("Please enter a question.")

        else:

            with st.spinner("Generating answer..."):

                # =========================
                # RETRIEVE DOCUMENTS
                # =========================

                retrieved_docs = retriever.invoke(query)

                # DEBUGGING
                st.subheader("Retrieved Chunks")

                for i, doc in enumerate(retrieved_docs):

                    st.write(f"### Chunk {i+1}")

                    st.write(doc.page_content)

                    st.divider()

                # =========================
                # BUILD CONTEXT
                # =========================

                context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs]
                )

                # =========================
                # PROMPT
                # =========================

                prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say:

"I could not find this information in the document."

Context:
{context}

Question:
{query}
"""

                # =========================
                # GENERATE RESPONSE
                # =========================

                response = llm.invoke(prompt)

                # =========================
                # SHOW ANSWER
                # =========================

                st.subheader("Answer")

                st.write(response)