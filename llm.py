from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re 

class LLM(): 
    def __init__(self, config): 
        #TODO: read all needed config from the yaml 
        llm = OllamaLLM(model=config['llm']["name"])
        self.chain = llm | StrOutputParser()
    
    def __call__(self, text:str): 
        return self.chain.invoke(text)


class VectorDB(): 
    def __init__(self, config):
        self.config = config   
        self.embedding_model = OllamaEmbeddings(model=self.config["vectordb"]["name"])
        self.vdb = Chroma(collection_name=self.config["vectordb"]["collection_name"],
                        embedding_function=self.embedding_model, 
                        persist_directory=self.config["vectordb"]["persist_directory"]
                        )
        self.name = self.config["vectordb"]["collection_name"]
    
    def add_document(self, document:str):
        existing_items = self.vdb.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing items: {len(existing_ids)}")
        
        docs = self._parse_documents(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config["vectordb"]["chunk_size"], 
                                                       chunk_overlap=self.config["vectordb"]["chunk_overlap"])
        chunks = text_splitter.split_documents(docs)
        #we tag each chunk with a unique id pased <source>:<page>:<chunk_num>
        chunks = self._gen_chunk_id(chunks)
        #we filter out the chunks that are already in the database
        chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
        if len(chunks) == 0:
            print("All chunks are already in the database")
            return
        return self.vdb.add_documents(chunks, ids=[chunk.metadata["id"] for chunk in chunks])
        

    def _parse_documents(self, documents:str):
        #TODO: replace this logic with Marked and markdown loader. 
        if documents.endswith(".pdf"):
            docs = PyPDFLoader(documents).load()
        elif documents.endswith(""):
            docs = PyPDFDirectoryLoader(documents).load()
        else: 
            raise ValueError(f"Document format not supported, {documents.split('.')[-1]}")
        return docs
    

    def _gen_chunk_id(self, chunks):
        page = ""
        source = ""
        for chunk in chunks:
            chunk_source = chunk.metadata.get("source")
            chunk_page = chunk.metadata.get("page")
            if f"{chunk_source}:{chunk_page}" != f"{source}:{page}":
                page = chunk_page
                source = chunk_source
                chunk_num = 0 
            chunk_id = f"{source}:{page}:{chunk_num}"
            chunk.metadata["id"] = chunk_id
            chunk_num += 1
        return chunks
            
        
    def search(self, query:str):
        retriver = self.vdb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': self.config["vectordb"]["score_threshold"], 'k': 6}
            )
        return retriver.invoke(query)


class Prompt(): 
    """
    Given a prompt in the format "abcd {arg1} efgh {arg2} ijkl",
    you can replace the args with the values using the call method.
    """
    def __init__(self, prompt:str):
        self.prompt = prompt
        self.kwargs = set(re.findall(r'\{([^{}]+)\}', prompt)) 

    def __call__(self, **kwargs):
        replacment = {} 
        for key in self.kwargs:
            if key not in kwargs.keys():
                raise ValueError(f"Missing key {key} in {kwargs.keys()}")
            value = kwargs.pop(key)
            replacment[key] = value
        if len(kwargs) > 0:
            raise ValueError(f"Extra keys {kwargs.keys()} in {self.kwargs}")
        self.replacements = replacment
        return re.sub(r'\{([^{}]+)\}', self._replace_match, self.prompt)

    def _replace_match(self, match):
        key = match.group(1).strip()
        return self.replacements[key]


#TODO: implement the class QueryRewarding
class QueryRewarding(): 
    
    def __init__(self, config):
        pass
    def __call__(self, *args, **kwds):
        pass