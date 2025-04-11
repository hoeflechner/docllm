import base64
import sys
import hashlib
import glob
import mimetypes
import logging
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough

import chromadb

load_dotenv()
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder')
OLLAMA_EMBEDDING_MODEL = os.getenv(
    'OLLAMA_EMBEDDING_MODEL', 'snowflake-arctic-embed2')
IMPORT_PATH = os.getenv('IMPORT_PATH', 'testfiles/')
ORIG_URL = os.getenv('ORIG_URL', 'testfiles/')

PROMPT=os.getenv('PROMPT',
                 """
                You are a helpful assistant who summarizes information and answers questions.
                You respond in the same language the question was asked 
                not in the language the context was given!
                If you don't know something, respond with 'I'm sorry, I don't know!'.
                Be structured, short but accurate.
                You get the information you need from the following context.
                """)

def embed(text):
    if st.session_state.get("embeddings",False):
        embeddings=st.session_state["embeddings"]
    else:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            temperature=0,
            base_url=OLLAMA_URL
        )
        st.session_state["embeddings"] = embeddings

    return embeddings.embed_query(text)

def db():
    if st.session_state.get("db",False):
        return st.session_state["db"]
    client = chromadb.Client(chromadb.Settings(is_persistent=True,
                                    persist_directory="chromadb",
                                ))
    coll = client.get_or_create_collection("documents")
    st.session_state["db"] = coll
    return coll

def get_md5(file):
    md5 = hashlib.md5(file.read()).hexdigest()
    # print(f"md5s of {file.name}: {md5}")
    return md5

def get_md5s_from_db():
    coll=db()
    data=coll.get()
    md5s= [m['md5'] for m in data['metadatas'] if m['md5'] is not None]
    return set(md5s)

def import_glob(glob_str):
    files = glob.glob(glob_str,recursive = True)
    import_files(files)

def describe(file_name):
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        verbose=True,
        base_url=OLLAMA_URL,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": 
             """
              describe the image in great detail. transcribe any text in the image.
             """},
            {
                "type": "image_url",
                "image_url": {"url": f"{file_name}"},
            },
        ],
    )
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(e)
        return ""

def import_files(files):
    for item in files:
        if isinstance(item, Path) or isinstance(item, str):
            if os.path.isfile(item):
                file = open(item, "rb")
            else:
                continue
        else:
            file = item

        print(f"loading {file.name}...")
        md5 = get_md5(file)
        if md5 in get_md5s_from_db():
            print("skipping - already loaded")
            continue

        text=""
        filetype = mimetypes.guess_type(file.name)[0]
        if filetype == 'text/plain':
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        if filetype == 'text/markdown':
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        if filetype == 'application/pdf':
            text = ""
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if isinstance(filetype,str) and filetype.startswith("image/"):
            text = ""
            text = describe(file.name)

        file.close()

        if text == "":
            print(f"skipping - could not read {filetype}")
            continue

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        # print(pdf)
        metadata = {"filename": file.name, "md5": md5}
        metadatas = [metadata for c in chunks]
        ids=[]
        for i in range(len(chunks)):
            ids.append(file.name+"/"+str(i))
        #ids= [str(hash(c)) for c in chunks]
        embeddings= [embed(c) for c in chunks]

        if  embeddings is None or len(embeddings) == 0:
            print(f"not text in {file.name}")
            continue

        db().upsert(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
        print(f"storing {md5}")
 
def get_response(user_query, chat_history):
    if "documents" not in st.session_state:
        st.session_state.documents = {}

    print(f"User input: {user_query}")

    data = db().query([embed(user_query)],n_results=3)
    for i,id in enumerate(data["ids"][0]):
        doc={"id":id, "page_content":data["documents"][0][i], "metadatas":data["metadatas"][0][i]}
        st.session_state.documents[id] = doc
    
    with st.sidebar:
        st.write("Sources")
        files=[doc['metadatas']['filename'] for doc in st.session_state.documents.values()]
        for file in set(files):
            file_name=file.replace(IMPORT_PATH,"")
            file_url=f"{ORIG_URL}/{file_name}"
            print(IMPORT_PATH)
            print(f"Source: {file_name} -> {file_url}")
            st.link_button(file_name, file_url)
                    
    context = ""

    for doc in st.session_state.documents.values():
        st.session_state.documents[doc["id"]] = doc
        context += doc["page_content"]+" \n"
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", PROMPT+" Context: {context}"              
            ),
            #("placeholder","{context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{user_query}")
        ]
    )
    
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        verbose=True,
        base_url=OLLAMA_URL,
    )

    chain = (
        prompt
        | llm 
        | StrOutputParser()
    )
    return chain.stream({
        "context": context,
        "chat_history": chat_history,
        "user_query": user_query,
    })

def main():
    st.set_page_config("Chat PDF")

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi!"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(get_response(
                user_query, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(content=response))
    
if __name__ == "__main__":
    for name, l in logging.root.manager.loggerDict.items():
        if "streamlit" in name:
            l.disabled = True
    if st.runtime.exists():
        main()
    else:
       import_glob(f"{IMPORT_PATH}/**")