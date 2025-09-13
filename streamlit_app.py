# Arithmetic Agent
# Uses a tool use pattern by using function tools
# Tool for addition
# Tool for subtraction
# Tool for multiplication
# Tool for division
# User provides a prompt such as "What is the sum of 1 and 2?" or "add 1 and 2" or "1 + 2" and "What is the product of 1 and 2?"
# The agent uses an LLM to decide which tool to execute and then executes it with the inputs and returns the results
# imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
import os
import streamlit as st
from pyalex import Works

@st.cache_resource(ttl="1d", show_spinner=False)
def createVectorStore():
    loader = CSVLoader(file_path="data.csv")
    data = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    vectorstore = Chroma.from_documents(
                        documents=data,                 # Data
                        embedding=embeddings,    # Embedding model
                        )
    return vectorstore
    #retriever = vectorstore.as_retriever()

# addition tool
@tool    # This identifies the following function as a tool to langgraph
# in the following statement the function name, the attributes their types and the output type are defined
def addition(x:int, y:int) -> int : 
    # The following docstring describes what the function can do and is used by the LLM to determine whethere this
    # is the tool to be called, and what are its inputs and outputs
    """
    This addition function adds two numbers and returns their sum. 
    It takes two integers as its inputs and the output is an integer.
    """
    return x + y

# subtraction tool
@tool
def subtraction(x:int, y:int) -> int : 
    """
    This subtraction function subtracts a number from another and returns the difference. 
    It takes two integers as its inputs and the output is an integer.
    """
    return x - y
# multiplication tool
@tool
def multiplication(x:int, y:int) -> int : 
    """
    This multiplication function multiplies two numbers returns the product. 
    It takes two integers as its inputs and the output is an integer.
    """
    return x * y
# division tool
@tool
def division(x:int, y:int) -> int : 
    """
    This division function divides one number by another and returns the quotient. 
    It takes two integers as its inputs and the output is an integer.
    """
    return x / y

@tool
def openAlex(term: str) -> str :
    """
    Searches for journal articles and books in the OpenAlex API using a keyword.
    """
    response = []
    if term:
        response = Works().search(term).get()
    else:
        response = "Could not extract search term"
    return response

@tool
def localDocs(query:str) -> str :
    """
    Searches for inforamtion about SJSU library and returns relevant documents
    """
    docs = vectorstore.similarity_search(query, k=4) # k specifies the number of documents to return
    return docs

st.title("Arithmetic Agent Chatbot")

# API key and endpoint configuration
# The preferred and secure approach is to store these in environment variables
# Do not share code with these values set
# os.environ["AZURE_OPENAI_API_KEY"] = ""
# os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["GOOGLE_API_KEY"] = st.secrets.key

# LLM configuration
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize st.session_state with the model
if "openai_model" not in st.session_state:
    st.session_state["azure_openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Model test
#test_yn = input("Do you want to test the model? (y/n) :")
# test_yn = "n"
# if test_yn == "y" :
#    prompt_query = input("Enter your question : ")
#    response = model.invoke(prompt_query)
#    print(response.content)

# Tool list
arithmeticagent_tools = [addition, subtraction, multiplication, division, openAlex, localDocs]

arithmeticagent_system_prompt = SystemMessage(
    """You are a math agent that can solve simple mathematics problems like addition, subtraction, multiplication and division. 
    Solve the mathematics problems provided by the user using only the available tools and not by yourself. Provide the answer given by the tool.  
    You are also able to take a natural language query and fetch and return a list of five relevant research articles with links from the OpenAlex API.
    You are also able to take a natural laanguage query about SJSU library and respond based on documents contained in the local database.
    """
)

debug_yn = True
debug_yn = st.checkbox("Debug")
arithmeticagent_graph = create_react_agent(
    model = model,
    state_modifier = arithmeticagent_system_prompt,
    tools = arithmeticagent_tools,
    debug = debug_yn)


detailed_yn = True
detailed_yn = st.checkbox("Detailed output")
if prompt := st.chat_input("Enter a simple mathematics question"): # if input has a value
    with st.chat_message("user"):         # set the role as user
        st.markdown(prompt)               # display the message

    # We add the message to the chat history with the role as user
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):    
        inputs = {"messages":[("user", prompt)]}
        result = arithmeticagent_graph.invoke(inputs)
        response = f"Agent response : {result['messages'][-1].content} \n"
        if detailed_yn:
            response = response + f"\nDetailed execution flow : \n"
            for message in result['messages']:
                response = response + "\n" + message.pretty_repr() + "\n"
        st_response = st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
