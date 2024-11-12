## Rag Q&A Conversation with PDF Including Chat history

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

hf_api_key = os.getenv("hf_api_key")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "RAG Document Q&A"

# Set up streamlit
st.set_page_config(page_title="Chat with PDF", page_icon="💬")

# Add Font Awesome stylesheet
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

# Title with icon
st.markdown('<h2><i class="fas fa-book"></i> Chat With PDF</h2>', unsafe_allow_html=True)

# Sidebar with an icon
# Instead of st.sidebar.title(), use st.sidebar.markdown() for HTML content
st.sidebar.markdown("<h3><i class='fas fa-upload'></i> Upload PDF</h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

# Input Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Check if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    # Statefully manage the chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Chat and history setup
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer the question."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the question in three sentences."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Customizing the background and chat design
        st.markdown(
            """
            <style>
            /* Unique background and modern styles */
            body {
                background: linear-gradient(135deg, #6A5ACD, #483D8B);
                font-family: 'Arial', sans-serif;
                color: white;
                margin: 0;
                padding: 0;
            }

            /* Chat message style */
            .chat-bubble {
                padding: 10px;
                border-radius: 15px;
                margin-bottom: 10px;
                width: 75%;
                animation: fadeIn 1s ease;
            }

            .user {
                background-color: #DCF8C6;
                align-self: flex-end;
                max-width: 75%;
            }

            .assistant {
                background-color: #E5E5EA;
                align-self: flex-start;
                max-width: 75%;
            }

            /* Chat container style - no scroll */
            .chat-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin-top: 20px;
                max-width: 800px;
                margin: 0 auto;
            }

            /* Input box customization */
            .input-container {
                display: flex;
                justify-content: space-between;
                width: 100%;
            }

            /* Add a smooth fade-in animation */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Style for the input box */
            .stTextInput>div>input {
                background-color: #E5E5EA;
                border-radius: 15px;
                padding: 10px;
                font-size: 16px;
                border: none;
            }

            .stTextInput>div>input:focus {
                outline: none;
                border: 2px solid #6A5ACD;
            }
            </style>
            """, unsafe_allow_html=True
        )

        # Chat interface and user input handling
        with st.container():
            # Display the chat messages without scroll
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            # User input field with an icon
            user_input = st.text_input("Your question: <i class='fa fa-question-circle'></i>", placeholder="Ask something about your PDF...", label_visibility="collapsed")

            # Process the user input and show responses
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )

                # Display chat bubbles inside the container
                chat_history = session_history.messages
                for message in chat_history:
                    if isinstance(message, HumanMessage):
                        st.markdown(f'<div class="chat-bubble user">**You:** {message.content}</div>', unsafe_allow_html=True)
                    elif isinstance(message, AIMessage):
                        st.markdown(f'<div class="chat-bubble assistant">**Assistant:** {message.content}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.sidebar.warning("Please enter the Groq API Key")
