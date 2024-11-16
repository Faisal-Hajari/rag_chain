FROM ubuntu
# Install python3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get install build-essential 
# Install python requirements
RUN pip install streamlit langchain langchain_ollama langchain_chroma langchain_community langchain_core pypdf langchain_openai

#create folder and copy files
RUN mkdir app 
WORKDIR /app
COPY . . 

# Expose port and run the app
WORKDIR /app/src
EXPOSE 8501
CMD ["streamlit", "run", "interface.py"]