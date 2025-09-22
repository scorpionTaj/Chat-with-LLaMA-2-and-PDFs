import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    pipeline,
)


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
        separator="\n", chunk_size=250, chunk_overlap=25, length_function=len
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

    embedding_model = st.session_state.selected_embedding

    # Load embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model, model_kwargs={"device": device}
    )
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

    model_name = st.session_state.selected_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "t5" in model_name.lower():
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map=device_map, dtype=torch.float16
        )
        pipeline_task = "text2text-generation"
    else:
        tokenizer.padding_side = "left"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="eager",
        )
        pipeline_task = "text-generation"

    # Create a Hugging Face pipeline
    llm_pipeline = pipeline(
        pipeline_task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=st.session_state.max_tokens,
    )

    # Wrap pipeline in LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Set up memory for conversational chain
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=2000, memory_key="chat_history", return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
    )


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

    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    # Display sources if available
    if "source_documents" in response and response["source_documents"]:
        with st.expander("📚 Sources"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.text(
                    doc.page_content[:500] + "..."
                    if len(doc.page_content) > 500
                    else doc.page_content
                )


def main():
    """
    Main function for the Streamlit application.
    """
    load_dotenv()
    st.set_page_config(
        page_title="Chat with LLaMA-2 and PDFs",
        page_icon="📚",
        layout="wide",
    )

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "microsoft/phi-3.5-mini-instruct"
    if "selected_embedding" not in st.session_state:
        st.session_state.selected_embedding = "sentence-transformers/all-MiniLM-L6-v2"
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 100
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = None

    # Main title
    st.title("📚 Chat with LLaMA-2 and PDFs")
    st.markdown(
        "Ask questions about your uploaded PDF documents using AI-powered conversation!"
    )

    # Create columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Chat interface
        st.subheader("💬 Conversation")

        # Chat input
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User message
                    with st.chat_message("user"):
                        st.write(message.content)
                else:  # Bot message
                    with st.chat_message("assistant"):
                        st.write(message.content)
        else:
            st.info("👋 Upload and process your PDFs, then start chatting!")

    with col2:
        # Status and controls
        st.subheader("📊 Status")
        if st.session_state.conversation:
            st.success("✅ Documents processed and ready!")
            if st.session_state.document_stats:
                st.json(st.session_state.document_stats)
        else:
            st.warning("⚠️ Please upload and process PDFs to start.")

        st.divider()

        # Reset buttons
        st.subheader("🔄 Reset")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = None
            st.rerun()

        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.document_stats = None
            st.rerun()

        st.divider()

        # Download chat history
        if st.session_state.chat_history:
            chat_text = "\n\n".join(
                [
                    f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                    for i, msg in enumerate(st.session_state.chat_history)
                ]
            )
            st.download_button(
                label="📥 Download Chat History",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Management")

        # Model Selection
        with st.expander("🤖 Model Settings"):
            st.markdown(
                "**Note:** Choose models based on your GPU VRAM. Smaller models (1.5B-3B) work best on 4GB GPUs."
            )
            st.session_state.selected_model = st.selectbox(
                "Select LLM Model:",
                [
                    "microsoft/phi-3.5-mini-instruct",
                    "microsoft/phi-3-mini-4k-instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "google/gemma-2-2b-it",
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "google/flan-t5-small",  # Keep as CPU fallback
                ],
                index=0,  # Default to first
            )
            st.session_state.selected_embedding = st.selectbox(
                "Select Embedding Model:",
                [
                    "BAAI/bge-small-en-v1.5",
                    "intfloat/e5-small-v2",
                    "thenlper/gte-small",
                    "nomic-ai/nomic-embed-text-v1.5",
                    "intfloat/multilingual-e5-small",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                index=5,  # Default to all-MiniLM
            )
            st.session_state.max_tokens = st.slider(
                "Max Response Tokens:", 50, 500, st.session_state.max_tokens
            )

        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True, type=["pdf"]
        )

        if pdf_docs:
            st.subheader("📋 Uploaded Files")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")

        if st.button("⚙️ Process Documents", use_container_width=True):
            with st.spinner("🔄 Processing documents..."):
                if not pdf_docs:
                    st.error("❌ Please upload at least one PDF file.")
                    return

                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("❌ No text found in the uploaded PDFs.")
                    return

                # Calculate stats
                total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
                total_words = len(raw_text.split())
                text_chunks = get_text_chunks(raw_text)
                total_chunks = len(text_chunks)

                st.session_state.document_stats = {
                    "Total PDFs": len(pdf_docs),
                    "Total Pages": total_pages,
                    "Total Words": total_words,
                    "Total Chunks": total_chunks,
                }

                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Documents processed successfully!")

        # Document Preview
        if pdf_docs and st.checkbox("Preview Extracted Text"):
            with st.expander("📖 Text Preview"):
                raw_text = get_pdf_text(pdf_docs)
                st.text_area(
                    "Extracted Text (first 1000 chars):", raw_text[:1000], height=200
                )

        st.divider()

        # Developer Info in expander
        with st.expander("👨‍💻 Developer Info"):
            st.markdown(
                """
            **Developer:** Tajeddine Bourhim
            **Location:** Meknes, Morocco 🇲🇦
            **LinkTree:** [My Social Media](https://linktr.ee/tajeddineofficiel)
            Built with ❤️ using **Streamlit**.
            """
            )


if __name__ == "__main__":
    main()
