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
import os
import requests
import streamlit as st
from pyalex import Works

@tool
def openAlex(term: str) -> str :
    """
    Searches for academic books in the OpenAlex API using a keyword.
    """
    response = []
    if term:
        response = Works().search(term).filter(type="book").get()
    else:
        response = "Could not extract search term"
    return response

@tool
def openLibrary (term:str) -> str :
    """
    Searches for books in the OpenLibrary API using a keyword.
    """
    response = []
    if term:
        params = {"q":term, 'sort':'new'}
        r = requests.get(url="https://openlibrary.org/search.json", params=params)
        response = r.json()
    else:
        response = "Could not extract search term"
    return response


st.title("AI-Powered Book Banner ✨")

# API key and endpoint configuration
# The preferred and secure approach is to store these in environment variables
# Do not share code with these values set
# os.environ["AZURE_OPENAI_API_KEY"] = ""
# os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["GOOGLE_API_KEY"] = st.secrets.key

# LLM configuration
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
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
bookbanagent_tools = [openAlex, openLibrary]

bookbanagent_system_prompt = SystemMessage(
    """
    You are a book banning agent that can identify book titles that should be removed from circulation because they violate content restrictions.
    Use your understanding of social norms to deduce keywords that match the user's intent.
    Always identify titles by searching the keyworkds you deduced. Do not generate the titles without using a tool.
    Identify titles to be banned by searching by keyword using the available tools. 
    Add a link for each book if one is available.
    For each book add an appropriate emoji that relates to the book's title.
    Do not repeat the same book title more than once.
    Do not include more than 20 books in the list.
    Begin your response by saying these are the books that meet the criteria for banning. Don't mention the emojis in the introduction.
    """
)

debug_yn = True
debug_yn = st.checkbox("Debug")
bookbanagent_graph = create_react_agent(
    model = model,
    state_modifier = bookbanagent_system_prompt,
    tools = bookbanagent_tools,
    debug = debug_yn)


detailed_yn = True
detailed_yn = st.checkbox("Detailed output")
if prompt := st.chat_input("Enter criteria for book banning"): # if input has a value
    with st.chat_message("user"):         # set the role as user
        st.markdown(prompt)               # display the message

    # We add the message to the chat history with the role as user
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):    
        inputs = {"messages":[("user", prompt)]}
        result = bookbanagent_graph.invoke(inputs)
        response = f"Agent response : {result['messages'][-1].content} \n"
        if detailed_yn:
            response = response + f"\nDetailed execution flow : \n"
            for message in result['messages']:
                response = response + "\n" + message.pretty_repr() + "\n"
        st_response = st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
