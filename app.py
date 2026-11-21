from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet
    Args:
    query: The query to search for
    Returns:
    The search result
    """

    print(f"Searching for {query}")
    return  tavily.search(query=query, max_results=3)


llm = ChatMistralAI(
    model="mistral-large-4",
    temperature=0,
    reasoning_effort="medium",
)

tools = [search]
agent = create_agent(model=llm, tools=tools)


result = agent.invoke(
    {"messages": [HumanMessage(
        content="What's the weather in Singapore today?")
    ]}
)

print(result)

# print(result["messages"][-1].text)
# print(result["messages"][-1].content[-1]["text"])