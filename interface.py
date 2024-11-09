import streamlit as st
import os

# Function to handle PDF upload
def upload_pdf(doc_path:str="docs"):
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
        # Generate assistant response
        response = responed(st.session_state.messages)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

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

# Main app
def main():
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
        tabs={"Upload PDF":upload_pdf, "Chatbot":chatbot},
        buttons={"Clear History":clear_history}
    )

    

if __name__ == "__main__":
    main()
