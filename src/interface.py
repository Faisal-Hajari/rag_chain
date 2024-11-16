import streamlit as st
import os
from functools import partial
from llm import Prompt, LLM, VectorDB
import yaml
import prompts
import logging 
import ops 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler to save logs to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Function to handle PDF upload
def upload_pdf(vdb:VectorDB, doc_path:str="docs"):
    st.title("Upload PDF File")
    uploaded_files = st.file_uploader("Choose a PDF file", 
                                     type="pdf", 
                                     accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Save the uploaded PDF file
            save_path = os.path.join(doc_path, uploaded_file.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
    vdb.add_document(doc_path)
    

# Function for the Chatbot
def chatbot(responed:callable= lambda x: "Hello!"):
    st.title("Chatbot")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream chat messages from the chatbot
        with st.chat_message("assistant"):
            response = "" 
            response_placeholder = st.empty()
            for chunk in responed(st.session_state.messages):
                response += chunk
                response_placeholder.markdown(response)

        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        logger.info(response)

def side_bar(
        tabs:dict[str:callable]={"Upload PDF":upload_pdf, "Chatbot":chatbot},
        buttons:dict[str:callable]={}
    ):
    st.sidebar.title("Navigation")

    #run buttons
    for name, func in buttons.items():
        if st.sidebar.button(name):
            func()

    #run selected tab
    app_tab = st.sidebar.radio("tab", tabs.keys())
    tabs[app_tab]()

   

def clear_history(): 
    st.session_state.messages = []

def read_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Main app
def main():
    conifg = read_config("config.yaml")
    llama = LLM(conifg)
    vdb = VectorDB(conifg)

    # Set up custom page config for smaller sidebar
    st.set_page_config(page_title="Streamlit App", layout="wide", initial_sidebar_state="expanded")
    # Custom CSS to reduce sidebar width
    st.markdown(
        """
        <style>
        /* Reduce sidebar width */
        .css-1d391kg {
            width: 150px;
        }
        .css-1d391kg .css-1q8dd3e {
            width: 150px;
        }
        .css-1d391kg .css-1v3fz7d {
            width: 150px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Run the sidebar
    side_bar(
        tabs={"Upload PDF":partial(upload_pdf, vdb), 
        "Chatbot":partial(chatbot,
                            partial(ops.rag_with_ref, llama, vdb)
                        )
        },
        buttons={"Clear History":clear_history}
    )

    

if __name__ == "__main__":
    main()
