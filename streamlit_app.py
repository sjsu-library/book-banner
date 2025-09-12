# Arithmetic Agent
# Uses a tool use pattern by using function tools
# Tool for addition
# Tool for subtraction
# Tool for multiplication
# Tool for division
# User provides a prompt such as "What is the sum of 1 and 2?" or "add 1 and 2" or "1 + 2" and "What is the product of 1 and 2?"
# The agent uses an LLM to decide which tool to execute and then executes it with the inputs and returns the results
# imports
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from getandrun_query import getandrun
from pc_query import pinecone_query
import streamlit as st

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

# database query tool
@tool
def db_query(natural_language_query):
    """
    This function takes a natural language text query and runs it against an
    employee table in a database and returns the result. 
    It takes a string as input and returns a list object. 
    """
    response = getandrun(natural_language_query)
    return response

# index query tool
@tool
def index_query(natural_language_query):
    """
    This function takes a natural language text query and runs it against an
    and index that has information about Vijai Gandikota and returns the result.
    It takes a string as input and returns a string object. 
    """
    response = pinecone_query(natural_language_query)
    return response

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
arithmeticagent_tools = [addition, subtraction, multiplication, division, db_query, index_query]

arithmeticagent_system_prompt = SystemMessage(
    """You are a math agent that can solve simple mathematics problems like addition, subtraction, multiplication and division. 
    Solve the mathematics problems provided by the user using only the available tools and not by yourself. Provide the answer given by the tool.  
    You are also able to take a natural language query and fetch answer from the employees table about employees. 
    Provide the answer given by the tool.
    You are also able to take a natural language query and fetch the answer from and index. Provide the answer given by the tool. 
    """
)

debug_yn = False
#debug_yn = st.checkbox("Debug")
arithmeticagent_graph = create_react_agent(
    model = model,
    state_modifier = arithmeticagent_system_prompt,
    tools = arithmeticagent_tools,
    debug = debug_yn)


detailed_yn = False
#detailed_yn = st.checkbox("Detailed output")
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
