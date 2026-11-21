from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

search_tool = TavilySearch()
tools = [search_tool]

llm = ChatMistralAI(
    model="mistral-large-4",
    temperature=0,
    reasoning_effort="medium",
)

agent = create_agent(model=llm, tools=tools)

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Search for 3 job postings for an AI engineer "
                    "in Singapore civil and public service on LinkedIn and list their details"
                )
            )
        ]
    }
)

print(result)