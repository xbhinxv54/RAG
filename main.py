import os
import bs4
import warnings
from langchain import hub
from typing import Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from typing_extensions import Annotated
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph,END
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter






warnings.filterwarnings('ignore')

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")





llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="input"
)
print("Loading embedding model...")
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  
# )
print("Loading documents...")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)



docs = loader.load()

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


total_documents = len(all_splits)
third = total_documents // 3
print("Assigning sections...")
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"



vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

# _ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")


class Search(TypedDict):
    query:Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question:str
    query:Search
    answer:str
    context:List[Document]


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query} #the query part of the state is updated with the structured query, query(Search) object is not returned

def retrieve(state:State):
    #retrieved_docs = vector_store.similarity_search(state["question"])
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
        k=3,
    )
    return {"context": retrieved_docs}


def generate(state:State):
    docs_content="\n\n".join([doc.page_content for doc in state["context"]])
    messages=prompt.invoke({"question": state["question"], "context": docs_content})
    reponse=llm.invoke(messages)    
    return {"answer": reponse.content}

graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_edge(START, "analyze_query")
graph_builder.add_edge("analyze_query", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

response = graph.invoke({"question": "What does the end of the post say about Task Decomposition?"})
print(response["answer"])