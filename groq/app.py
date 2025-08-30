import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )

st.title("ChatGroq Demo")

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

prompt = ChatPromptTemplate.from_template("""
Answer the question as truthfully as possible using the provided context. 
If the answer is not contained within the context, say "I don't know".

Context:
{context}

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Input your question here")

if user_prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input": user_prompt})
    st.write("Response time:", time.process_time() - start)
    st.write(response["answer"])

    with st.expander("Document Similarity Search Results"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------------")
