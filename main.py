from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer",
    )


search_tool = TavilySearch()

llm = ChatMistralAI(
    model="mistral-large-4",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[search_tool],
    response_format=ToolStrategy(AgentResponse),
)


def main():
    print(f"Using built-in Tavily LangChain tool")
    print(f"Structured output with LangChain agents using Pydantic\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search for 2 job postings for an AI engineer "
                        "in Singapore civil and public service on LinkedIn and list their details"
                    )
                )
            ]
        }
    )

    response: AgentResponse = result["structured_response"]
    print(response.answer)

    print("\nSources:")
    for i, s in enumerate(response.sources, 1):
        print(f"  {i}. {s.url}")

    # print(type(result["structured_response"]))
    # print(result["structured_response"])


if __name__ == "__main__":
    main()