import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from htmlTemplates import css, bot_template, user_template


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    """
    Reads text from a list of PDF files.
    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Combined text content from all PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():  # Handle cases where text extraction may fail
                text += page.extract_text()
    return text


# Split text into manageable chunks
def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for processing.

    Args:
        text (str): The input text to split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# Create a vector store for text chunks
# Create a vector store for text chunks
def get_vectorstore(text_chunks):
    """
    Creates a FAISS vector store for semantic search.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        FAISS: A FAISS vector store.
    """

    # Choose the embedding model based on hardware capability
    # -------------------------------------------------------
    # 1. CPU-Optimized Embedding Model
    #    Example: "sentence-transformers/all-MiniLM-L6-v2"
    #    Lightweight embedding model designed for CPU environments.
    # -------------------------------------------------------
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # -------------------------------------------------------
    # 2. Medium GPU Embedding Model
    #    Example: "hkunlp/instructor-large"
    #    More advanced model requiring a moderate GPU setup (e.g., NVIDIA T4).
    # -------------------------------------------------------
    # embedding_model = "hkunlp/instructor-large"

    # -------------------------------------------------------
    # 3. High-End GPU Embedding Model
    #    Example: "hkunlp/instructor-xl"
    #    High-performance model designed for advanced GPUs (24 GB VRAM or more).
    # -------------------------------------------------------
    # embedding_model = "hkunlp/instructor-xl"

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# Set up the conversational chain
def get_conversation_chain(vectorstore):
    """
    Sets up a conversational retrieval chain using the LLaMA-2 model.

    Args:
        vectorstore (FAISS): The vector store for document retrieval.

    Returns:
        ConversationalRetrievalChain: A LangChain conversation chain.
    """

     # Choose the model based on hardware capability
    # -------------------------------------------------------
    # 1. CPU-Optimized Model: google/flan-t5-small
    #    Lightweight model suitable for CPU-based environments.
    #    Uses significantly less memory and works on slower hardware.
    #    Example: "google/flan-t5-small" (77M parameters)
    # -------------------------------------------------------
    # model_name = "google/flan-t5-small"

    # -------------------------------------------------------
    # 2. Medium GPU Model: meta-llama/Llama-2-7b-chat-hf
    #    A robust model requiring medium GPU capability.
    #    Works well on GPUs like NVIDIA T4 or 8-16 GB GPUs.
    #    Example: "meta-llama/Llama-2-7b-chat-hf" (7B parameters)
    # -------------------------------------------------------
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # -------------------------------------------------------
    # 3. High-End GPU Model: meta-llama/Llama-2-13b-chat-hf
    #    Requires a high-performance GPU (24 GB VRAM or more).
    #    Example GPUs: A100, RTX 3090, or RTX 4090.
    #    Example: "meta-llama/Llama-2-13b-chat-hf" (13B parameters)
    # -------------------------------------------------------
    # model_name = "meta-llama/Llama-2-13b-chat-hf"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="float32")

    # Create a Hugging Face pipeline
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    # Wrap pipeline in LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Set up memory for conversational chain
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# Handle user input and update chat history
def handle_userinput(user_question):
    """
    Handles the user's input and generates a response using the conversation chain.

    Args:
        user_question (str): The user's question.

    Updates:
        st.session_state.chat_history (list): The updated chat history.
    """
    if not st.session_state.conversation:
        st.warning("Please process documents first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the chat messages
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot message
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function for Streamlit app
def main():
    """
    Main function for the Streamlit application.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with LLaMA-2 and PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # Apply custom CSS

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # App Header
    st.header("Chat with LLaMA-2 and PDFs :books:")

    # User input for questions
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    # Sidebar Developer Info
    st.sidebar.title("Developer Info")
    st.sidebar.info("""
    **Developer:** Tajeddine Bourhim

    **Location:** Meknes, Morocco 🇲🇦

    **LinkTree:** [My Social Media](https://linktr.ee/tajeddineofficiel)

    This app is built with ❤️ using **Streamlit**.
    """)

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                    return


                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully! You can now ask questions.")

if __name__ == '__main__':
    main()
