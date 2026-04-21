import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Context-Aware RAG System", page_icon="📚", layout="wide")
st.title("📚 Context-Aware RAG System")
st.caption("Chat with Turkish financial report PDFs using a history-aware RAG pipeline.")

VECTORSTORE_DIR = Path("vectorstore/db_faiss")

def model_hf_hub(model: str = "meta-llama/Meta-Llama-3-8B-Instruct", temperature: float = 0.1):
    return HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        max_new_tokens=512,
        return_full_text=False,
        task="text-generation",
    )

def model_openai(model: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(model=model, temperature=temperature)

def model_ollama(model: str = "phi3", temperature: float = 0.1):
    return ChatOllama(model=model, temperature=temperature)

def get_llm(model_class: str):
    if model_class == "hf_hub":
        llm = model_hf_hub()
        token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return llm, token_s, token_e
    if model_class == "openai":
        return model_openai(), "", ""
    if model_class == "ollama":
        return model_ollama(), "", ""
    raise ValueError(f"Unknown model class: {model_class}")

def config_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})

def config_rag_chain(model_class: str, retriever):
    llm, token_s, token_e = get_llm(model_class)

    context_q_system_prompt = (
        "Given the following chat history and a follow-up question, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do not answer the question. Only reformulate it when necessary."
    )
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", token_s + context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}" + token_e),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_q_prompt,
    )

    qa_prompt_template = """You are a financial-report analysis assistant.
Answer the user's question only using the retrieved context from the uploaded PDF documents.

Rules:
- Answer in Turkish unless the user asks for another language.
- If the answer is not supported by the context, say: "Belgede belirtilmemiştir."
- Be concise and accurate.
- When relevant, keep numbers, percentages, dates, and units exactly as written in the source.
- If the answer refers to a note, page, or section, mention it briefly.

Question: {input}

Context:
{context}
"""
    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)

def uploaded_file_names(uploads):
    return tuple(sorted(f.name for f in uploads)) if uploads else tuple()

with st.sidebar:
    st.header("Configuration")
    model_class = st.selectbox(
        "LLM provider",
        options=["openai", "hf_hub", "ollama"],
        index=0,
        help="OpenAI is easiest. Hugging Face and Ollama are alternatives.",
    )
    uploads = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Upload annual reports or financial statements")
    st.markdown("- Ask specific, source-grounded questions")
    st.markdown("- Follow-up questions use chat history")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Merhaba! PDF belgeleriniz hakkında soru sorabilirsiniz."),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)

if not uploads:
    st.info("Başlamak için kenar çubuğundan en az bir PDF yükleyin.")
    st.stop()

user_query = st.chat_input("Sorunuzu yazın...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        start = time.time()

        current_docs = uploaded_file_names(uploads)
        if st.session_state.docs_list != current_docs:
            st.session_state.docs_list = current_docs
            with st.spinner("Belgeler işleniyor ve vektör indeksi oluşturuluyor..."):
                st.session_state.retriever = config_retriever(uploads)

        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        with st.spinner("Yanıt hazırlanıyor..."):
            result = rag_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": st.session_state.chat_history[:-1],
                }
            )

        response = result["answer"]
        st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))

        sources = result.get("context", [])
        st.session_state.last_sources = sources

        if sources:
            st.markdown("#### Kaynaklar")
            for idx, doc in enumerate(sources, start=1):
                source = os.path.basename(doc.metadata.get("source", "document.pdf"))
                page = doc.metadata.get("page", "Belirtilmemiş")
                ref = f"🔗 Kaynak {idx}: *{source} - s. {page}*"
                with st.popover(ref):
                    st.caption(doc.page_content[:1500])

        elapsed = time.time() - start
        st.caption(f"Yanıt süresi: {elapsed:.2f} saniye")
