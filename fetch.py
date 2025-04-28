from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlite3
from langchain_core.documents import Document


class retrive_rag:

    def reciprocal_rank_fusion(self, result: list[list], k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}
            # Iterate through each list of ranked documents
            for docs in result:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # print(doc)
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            # print(reranked_results)
            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results




    def query_with_index(self, student_name):
            """Query the Students table using an index on the name column."""
            with sqlite3.connect('my_database.db') as connection:
                cursor = connection.cursor()

                # SQL command to select a student by name
                select_query = 'SELECT * FROM Datastore WHERE id = ?;'


                # Execute the query with the provided student name
                cursor.execute(select_query, (student_name,))
                result = cursor.fetchall()  # Fetch all results

                # Display results
                # print(f"Query result: {result}")

            return result



    def main(self, question):
        embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False,  openai_api_key="sk-1234", base_url="http://localhost:8082/v1",model="mxbai-embed-large-v1-f16")

        persist_directory = "chroma_db" 

        loaded_vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory) 

        retriever=loaded_vectordb.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(base_url="http://localhost:8081/v1",model="gemma-2-2b-it.Q6_K", api_key="LM")

        
        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )



        retrieval_chain_rag_fusion = generate_queries | retriever.map() | self.reciprocal_rank_fusion
        docs = retrieval_chain_rag_fusion.invoke({"question": question})



        
        
        textdb=[]

        for i in range(int(len(docs)/4)):
            ind=self.query_with_index(docs[i][0].metadata['id'])
            text=Document(
                page_content=ind[0][1],
            )
            textdb.append(text)

        return textdb


