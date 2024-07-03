#!/usr/bin/env python
import os
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

prompt_react = hub.pull("hwchase17/react")
os.environ["TAVILY_API_KEY"] = "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"

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
    
calculatorTool = Tool(
    name=f"calculator",
    func=calculator,
    description=f"a calculator tool",
)
search = TavilySearchResults(max_results=1)
tools = [search, calculatorTool]
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_path = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = LlamaCpp(model_path=model_path,
                 temperature=0.75,
                 max_tokens=2000,
                 top_p=1,
                 callback_manager=callback_manager,
                 verbose=True)
agent = create_react_agent(model, tools=tools, prompt=prompt_react)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
result = agent_executor.invoke({"input": "10 x 29 = ?"})
print(result)