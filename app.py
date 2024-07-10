import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
import ssl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the GROQ_API_KEY from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit configuration
st.set_page_config(page_title="Langchain: Summarize text from YouTube or website", layout="centered")
st.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .stButton button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
    }
    .spinner {
        border: 16px solid #f3f3f3;
        border-top: 16px solid #000000;
        border-radius: 50%;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Langchain: Summarize text from YouTube or website")
st.subheader('Summarize URL')

# Input for the URL
generic_url = st.text_input("Enter the URL (YouTube or website) to summarize", value='')

def summarize_content(generic_url):
    if not generic_url.strip():
        st.error("Please provide a URL to get started")
    elif not validators.url(generic_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                context = ssl.create_default_context()
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=context,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                data = loader.load()

                # Initialize the language model
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

                # Create the prompt template
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template='''Provide a summary of the following content in 300 words:
                    Content: {text}'''
                )

                # Create and run the summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run({"input_documents": data})
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display button and handle click
if st.button("Summarize the content"):
    summarize_content(generic_url)

# Footer
st.markdown(
    """
    <div class="footer">
        Maintained and developed by <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: white;">Kshitiz Garg</a>
    </div>
    """,
    unsafe_allow_html=True
)
