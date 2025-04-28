from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
import os
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from typing import List
from typing_extensions import TypedDict
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from fetch import retrive_rag
from conv_memory import memory

os.environ["GOOGLE_CSE_ID"] = "957fb9e872df34d4a"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAij6HMt0mnWKuiWPwIzkILGMYRAB07OFY"

class reflection:
    def main(self, question):
        llm = ChatOpenAI(base_url="http://localhost:8081/v1",model="gemma-2-2b-it.Q6_K.gguf", api_key="LM", temperature=0)
        llm1 = ChatOpenAI(base_url="http://localhost:8081/v1",model="gemma-2-2b-it.Q6_K.gguf", api_key="LM", temperature=0.5)
        a=retrive_rag()

        template = """You are a helpful assistant. Use the following pieces of context to answer the question, answer should be five to six lines. If you cannot find the answer in the context, just say "I don't know." Do not make up an answer.

        Context: {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            prompt
            | llm1
            | StrOutputParser()
        )


        prompt = PromptTemplate(
            template="""You are an expert at routing a user question to a vectorstore or web search. \n
            Use the vectorstore for questions on Python language documentation \n
            You do not need to be stringent with the keywords in the question related to these topics. \n
            Otherwise, use keyword 'web-search'. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
            Return the a JSON with a single key 'datasource' and no premable or explanation. \n
            Question to route: {question}""",
            input_variables=["question"],
        )

        question_router = prompt | llm | JsonOutputParser()



        ### Hallucination Grader

        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by some of the facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "documents"],
        )

        hallucination_grader = prompt | llm | JsonOutputParser()



        ### Answer Grader
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}
            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "question"],
        )

        answer_grader = prompt | llm | JsonOutputParser()



        # Question transformation
        prompt = PromptTemplate(
            template="""
            Given the summary of a previous conversation and a follow up question, rephrase the follow up question to be a standalone question, in the same language as the original question.
            Chat History:
            {summary}
            Follow Up Input: {question}
            Standalone question:""",
            input_variables=["summary", "question"],
        )

        question_transform = prompt | llm | StrOutputParser()



        # previous conv relation
        prompt = PromptTemplate(
            template="""Given the following summary of a previous conversation and a user's question, determine if the question is relevant to the summary.

            Summary:
            {summary}

            Question:
            {question}

            Give a binary score 'yes' or 'no' to indicate whether the question is related to the summary. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            """,
            input_variables=["summary", "question"],
        )

        transform_dec = prompt | llm | JsonOutputParser()



        ### Google Search Tool
        search = GoogleSearchAPIWrapper()
        googlesearch = Tool(
            name="google_search",
            description="Search Google for results about anything.",
            func=search.run,
        )



        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                documents: list of documents
            """

            question: str
            generation: str
            documents: List[str]
            summary: str
            counter: int

        ## Define the nodes
        def initialize(state):
            """
            Initialize the graph state

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, question, that contains document and summary
            """
            print("---INITIALIZE---")
            mem=memory()
            summary = mem.get_memory()
            return {"summary": summary, "documents": [], "counter": 0} 
        

        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            print("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            # documents = retriever.get_relevant_documents(question)
            documents=a.main(question)
            print("documents:", documents)
            return {"documents": documents, "question": question}


        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, counter, generation, that contains LLM generation
            """
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]
            state["generation"] = ""
            counter = state["counter"]
            counter += 1
            # RAG generation
            generation = final_rag_chain.invoke({"context": documents, "question": question})
            # print("generation: ", generation)
            return {"documents": documents, "question": question, "generation": generation, "counter": counter}


        def web_search(state):
            """
            Web search based on the re-phrased question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with appended web results
            """
            print("---WEB SEARCH---")
            question = state["question"]
            documents = state["documents"]
            web_results = googlesearch.invoke({"query": question})

            documents.extend(
                [
                    Document(page_content=web_results)
                ]
            )
            print("documents", documents)
            return {"documents": documents, "question": question}




        def summarize_conversation(state):

            summary = state["summary"]
            generation = state["generation"]
            mem=memory()
            print(summary)

            if summary:

                summary_message = """
                    f"This is a summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages below:\n\n"
                    {generation}\n\n"
                    Should not excede 50 words. \n\n
                """

            else:
                summary_message = """{generation}\n\n
                    Create a short summary of the conversation above:{summary}
                """

            prompt = ChatPromptTemplate.from_template(summary_message)
            messages = prompt | llm | StrOutputParser()
            response = messages.invoke({"generation": generation, "summary": summary})
            mem.add_memory(response)
            return {"generation": generation, "summary": summary}


        def transform_query(state):
            """
            Transform the question to a standalone question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, question, that contains the transformed question
            """
            print("---TRANSFORM QUERY---")
            question = state["question"]
            summary = state["summary"]
            ans=question_transform.invoke({"summary": summary, "question": question})
            print("transformed question:", ans)
            return {"summary": summary, "question": ans}
        

        def route_questionn(state):
            question = state["question"]
            return {"question": question}


        ### Edges

        def route_if_relevant(state):
            """
            Checks if a question is relevant to a conversation summary using LangChain.

            Args:
                summary (str): The summary of the previous conversation.
                question (str): The user's question.

            Returns:
                str: A string indicating whether the question is relevant.
            """
            summary = state["summary"]
            question = state["question"]
            route=transform_dec.invoke({"summary": summary,"question": question})
            grade = route["score"]
            if grade == "yes":
                print("---DECISION: RELATED TO PREVIOUS CONVERSATIONS---")
                return "Yes"
            else:
                print("---DECISION: NO RELATION TO PREVIOUS CONVERSATIONS---")
                return "No"


        def route_question(state):
            """
            Route question to web search or RAG.

            Args:
                state (dict): The current graph state

            Returns:
                str: Next node to call
            """
            print("---ROUTE QUESTION---")
            question = state["question"]
            source = question_router.invoke({"question": question})
            print("source:", source)
            if source["datasource"] == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif source["datasource"] == "vectorstore":
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"



        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """

            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]
            counter = state["counter"]
            if counter > 1:
                print("---DECISION: MAXIMUM NUMBER OF ATTEMPTS REACHED---")
                return "useful"
            
            score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score["score"]

            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                print("---GRADE GENERATION vs QUESTION---")
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score["score"]
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"
            





        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("initialize", initialize)  # initialize
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("web_search", web_search)  # web search
        workflow.add_node("generate", generate)  # generate
        workflow.add_node("summarize_conversation", summarize_conversation)  # summarize_conversation
        workflow.add_node("transform_query", transform_query)  # question_transform
        workflow.add_node("route_questionn", route_questionn)  # route_question



        # Build graph
        workflow.add_edge(START, "initialize")

        workflow.add_conditional_edges(
            "initialize",
            route_if_relevant,
            {
                "Yes": "transform_query",
                "No": "route_questionn",
            },
        )

        workflow.add_edge("transform_query", "route_questionn")
        workflow.add_conditional_edges(
            "route_questionn",
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "generate")

        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "not useful": "web_search",
                "useful": "summarize_conversation",
            },
        )
        workflow.add_edge("summarize_conversation", END) 


        # Compile
        app = workflow.compile()


        # Run
        inputs = {"question": question, "documents": []}
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
            pprint("\n---\n")

            # Final generation
        return(value["generation"])