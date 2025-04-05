import os
import bs4
import warnings
from langchain import hub
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




warnings.filterwarnings('ignore')

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")





llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="input"
)

model_kwargs = {'device': 'cuda'} 
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)



docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


_ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")
print(prompt)