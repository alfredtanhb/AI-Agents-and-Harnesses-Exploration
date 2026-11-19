from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()


def main():
    information = """
    Mistral Large 4, codenamed Le Chonk, is Mistral AI’s flagship open-weight multimodal Mixture-of-Experts (MoE) model. 
    Comprising roughly 1 trillion total parameters with 49 billion active per token, it pairs a dense initial decoder 
    layer with 71 MoE layers—routing tokens across four of 128 specialized experts plus one shared expert. Built natively 
    for multimodality, it integrates a 1.6B Pixtral vision encoder and shares key-value (KV) states across layers to 
    efficiently support an expansive 1-million-token context window.

    Trained from scratch on 3,800 NVIDIA Grace Blackwell GPUs across more than 160 languages, Le Chonk excels in agentic 
    coding, enterprise workflows, and defensive cybersecurity. It leads open-weight benchmarks on Terminal-Bench 4.0 and 
    Cybench while demonstrating 97.1% attack resistance on the B3 Agent Security Benchmark. To balance latency and 
    compute costs, developers can dynamically toggle between low, medium, and high reasoning modes.

    Distributed under the Mistral Ecosystem Open-Weight License, the model enables data-sovereign, self-hosted deployments. 
    It ships with day-zero support across Transformers, vLLM, and llama.cpp, offering FP4 and 2-bit quantizations that 
    scale from multi-GPU clusters down to unified-memory workstations.
    """

    summary_template = """
    Given the information {information} about AI model, I want you to create:
    1. A short summary
    2. Two interesting facts about the AI model
    3. Provide proper paragraphs, indentation, bullets or point form.

    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatMistralAI(
    model="mistral-large-4",
    temperature=0,
    model_kwargs={"reasoning_effort": "medium"},  # Options: "low", "medium", "high"
    )
    

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})

    # print(response.content[-1]["text"])

    if isinstance(response.content, list):
        text = "".join(
            block["text"] for block in response.content if block.get("type") == "text"
        )
    else:
        text = response.content

    print(text)
    

if __name__ == "__main__":
    main()


