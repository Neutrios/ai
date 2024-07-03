import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
    OnlinePDFLoader,
    UnstructuredFileLoader,
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.tools.retriever import create_retriever_tool
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, Tool, create_react_agent

# uncomment this if nltk error

# import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()

prompt_react = hub.pull("hwchase17/react")
@st.cache_resource
def load_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model_path = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model = LlamaCpp(model_path=model_path,
                    temperature=0.75,
                    max_tokens=2000,
                    top_p=1,
                    callback_manager=callback_manager,
                    verbose=True)
    return model

@st.cache_resource
def load_vectorstore():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    #pdf
    loader = PyPDFDirectoryLoader("data/")
    data = loader.load()
    docs_list.extend(data)
    loader = DirectoryLoader('data/', glob="**/*.txt")
    docs = loader.load()
    docs_list.extend(data)
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

@st.cache_resource
def load_tavily_search():
    os.environ["TAVILY_API_KEY"] = "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"
    return TavilySearchResults(max_results=1)

def calculator(x, y, op) -> float:
    if(op == "+"):
        return x + y
    elif(op == "-"):
        return x - y
    elif(op == "*"):
        return x * y
    elif(op == "/"):
        return x / y
    else:
        return "Invalid operator"
    
def main():
    st.title("GPT4All Chatbot")
    llm = load_model()
    calculatorTool = Tool(
        name=f"calculator",
        func=calculator,
        description=f"a calculator tool",
    )
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    )

    # initialize the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You are here to provide information and answer questions.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Construct the Tools agent
    search_tool = load_tavily_search()
    tools = [search_tool,calculatorTool]
    agent = create_react_agent(llm, tools=tools, prompt=prompt_react)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    qa_chain = qa | llm


    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        try:
            print(query)
            result = qa_chain.invoke({"query": query})
            print(result)
        except Exception as e:
            print(e)
    elif st.button("Execute"):
        try:
            result = agent_executor.invoke({"input": query})
            print(result)
        except Exception as e:
            print(e)
if __name__ == "__main__":
    main()
