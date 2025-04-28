import os
import pdf4llm
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain.load import dumps, loads
import sqlite3

persist_directory = "chroma_db" 
# Specify the directory path
folder_path = r'C:\Users\Jayden S G\Documents\A\langchain\data'

# Get the list of all files and directories in the specified folder
files_and_directories = os.listdir(folder_path)

# Filter out only the files
files = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]


md_text={}
for index,file in enumerate(files):
    pdf_name=pdf_name=folder_path+'/'+file
    md_text[index] = pdf4llm.to_markdown(pdf_name)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
md_header_splits=[]
for i in range(len(md_text)):
    markdown_document = md_text[i]
    sp=markdown_splitter.split_text(markdown_document)
    md_header_splits.append(sp)

embed_list=[]
textdb=[]
length_of_index=0
for i in range(len(md_header_splits)):
    length_of_index += len(md_header_splits[i])

for i in range(len(md_header_splits)):
    h1=""
    h2=""
    h3=""
    h4=""
    h5=""
    h6=""
    h1=md_header_splits[i][0].metadata["Header 1"]
    for id in range(len(md_header_splits[i])):

        try:
            h2=md_header_splits[i][id].metadata["Header 2"]
            h3=""
            h4=""
            h5=""
            h6=""
        except KeyError:
            h2

        try:
            h3=md_header_splits[i][id].metadata["Header 3"]
            h4=""
            h5=""
            h6=""
        except KeyError:
            h3

        try:
            h4=md_header_splits[i][id].metadata["Header 4"]
            h5=""
            h6=""
        except KeyError:
            h4

        try:
            h5=md_header_splits[i][id].metadata["Header 5"]
            h6=""
        except KeyError:
            h5

        try:
            h6=md_header_splits[i][id].metadata["Header 6"]
        except KeyError:
            h6
        


        emb=Document(
            page_content=h1+"-"+h2+"-"+h3+"-"+h4+"-"+h5+"-"+h6,
            metadata={"id":h1+str(id)},
        )

        text=Document(
            page_content=md_header_splits[i][id].page_content,
            metadata={"id":h1+str(id)},
        )


        textdb.append(text)
        embed_list.append(emb)

embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False,  openai_api_key="sk-1234", base_url="http://localhost:8082/v1",model="mxbai-embed-large-v1-f16")
vectorstore = Chroma.from_documents(documents=embed_list, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)


with sqlite3.connect('my_database.db') as connection:
    table_name = 'Datastore'
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    result = cursor.fetchone()

    if result:
        # Table exists, so we can proceed with the insert operation
        insert_query = '''
        INSERT INTO Datastore (id, text) 
        VALUES (?, ?);
        '''
        students_data = [(textdb[i].metadata["id"], textdb[i].page_content) for i in range(length_of_index)]

        # Execute the query for multiple records
        cursor.executemany(insert_query, students_data)

        # Commit the changes
        connection.commit()
    else:
        # Table does not exist, so we need to create it first
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS Datastore (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL
        )
        '''

        # Execute the SQL command
        cursor.execute(create_table_query)

        # Commit the changes
        connection.commit()


        insert_query = '''
        INSERT INTO Datastore (id, text) 
        VALUES (?, ?);
        '''
        students_data = [(textdb[i].metadata["id"], textdb[i].page_content) for i in range(length_of_index)]

        # Execute the query for multiple records
        cursor.executemany(insert_query, students_data)

        # Commit the changes
        connection.commit()
