import json
import os
import requests
import streamlit as st

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pyalex import Works


# Configure API key
os.environ["GOOGLE_API_KEY"] = st.secrets.key


@tool
def openAlex(term: str) -> str:
    """Search OpenAlex for academic books matching a keyword."""
    if not term:
        return "Could not extract a search term."

    results = Works().search(term).filter(type="book").get()
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def openLibrary(term: str) -> str:
    """Search OpenLibrary for books matching a keyword."""
    if not term:
        return "Could not extract a search term."

    response = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": term, "sort": "new"},
        timeout=30,
    )
    response.raise_for_status()

    return json.dumps(response.json(), ensure_ascii=False)


st.title("AI-Powered Book Banner ✨")


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=2,
)


system_prompt = """
You are a book banning research agent.

Use the available search tools to identify books matching the user's criteria.
Do not invent book titles without searching for them first.

If the user's criteria are unclear, ask a follow-up question and provide examples
of possible search keywords.

For each result:
- Include the book title.
- Include a link when one is available.
- Add an appropriate emoji.
- Do not repeat titles.
- Return no more than 20 books.

Begin the response by saying:
"These are the books that meet the criteria for banning:"
"""


agent = create_react_agent(
    model=model,
    tools=[openAlex, openLibrary],
    prompt=system_prompt,
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


detailed_output = st.checkbox("Detailed output")


if prompt := st.chat_input("Enter criteria for book banning"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = agent.invoke(
            {"messages": [("user", prompt)]}
        )

        final_message = result["messages"][-1]
        response = final_message.content

        final_message = result["messages"][-1]
        content = final_message.content

    if isinstance(content, list):
        response = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    else:
        response = str(content)

    if detailed_output:
        response += "\n\n### Detailed execution flow\n"
        for message in result["messages"]:
            response += f"\n{message.pretty_repr()}\n"

    st.markdown(response)


    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )