from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import time

class memory:
    def add_memory(self, summary):
        
        persist_directory = "conv_memory_db"

        embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False, openai_api_key="sk-1234", base_url="http://localhost:8082/v1",model="mxbai-embed-large-v1-f16")

        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

        # Get the current time
        current_time = time.time()
        
        vectorstore.delete(ids="1")

        me=Document(
            page_content=summary,
            metadata={"timestamp":current_time},
        )

        
        vectorstore = Chroma.from_documents(documents=[me], 
                                            embedding=embeddings,
                                            persist_directory=persist_directory,
                                            ids="1")
    
        



    def get_memory(self):
        persist_directory = "conv_memory_db"
        
        embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False,  openai_api_key="sk-1234", base_url="http://localhost:8082/v1",model="mxbai-embed-large-v1-f16")

        loaded_vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory) 

        fetched_memory = loaded_vectordb.get(ids="1")

        prev_conv_time=fetched_memory['metadatas'][0]['timestamp']

        # Get the current time
        current_time = time.time()
         
        if prev_conv_time<current_time-86400:
            loaded_vectordb.delete(ids="1")
            return ""
            
        return fetched_memory['documents']