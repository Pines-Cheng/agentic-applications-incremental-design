# the ReAct pattern

- prerequisites from `README.md`
- `python3 -m venv venv && source venv/bin/activate`
- create a `requirements.txt` file with the following content:

```
langchain-core
langchain-ollama
langgraph
ollama
pydantic
```

- `pip install -r requirements.txt`
- let's use the tools we've created in the previous example:

```python
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import random

text_model = ChatOllama(model="llama3.1")


@tool
def get_top_gainers_and_losers() -> str:
    """Get today's top gainers and losers in the market.

    Use this tool to get the top gainers and losers in the market.
    """
    return f"This is a mock result for top losers and winners, assume it is a relevant result as we are in a testing environment."


@tool
def forecast_stock_price(query: str) -> str:
    """Forecast the stock price of a given company.

    Use this tool to forecast the stock price of a given company.
    """
    return f"${random.randint(1, 5000)}."


tools = [get_top_gainers_and_losers, forecast_stock_price]

text_model_with_tools = text_model.bind_tools(tools)
```

- ... and then initialize our main agent node:

```python
from langgraph.graph import MessagesState

sys_msg = SystemMessage(
    content="""You are an investment analyst equipped with tools such as stock price forecasting and getting top gainers and losers in the market. 
If using tools is not relevant to the user's question, just return a general answer without using tools."""
)

def agent(state: MessagesState):
    return {"messages": [text_model_with_tools.invoke([sys_msg] + state["messages"])]}
```

- we are now ready to build our graph:

```python
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
)
builder.add_edge("tools", "agent")
graph = builder.compile()
```

- your full script should look like this:

```python
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import random

text_model = ChatOllama(model="llama3.1")


@tool
def get_top_gainers_and_losers() -> str:
    """Get today's top gainers and losers in the market.

    Use this tool to get the top gainers and losers in the market.
    """
    return f"This is a mock result for top losers and winners, assume it is a relevant result as we are in a testing environment."


@tool
def forecast_stock_price(query: str) -> str:
    """Forecast the stock price of a given company.

    Use this tool to forecast the stock price of a given company.
    """
    return f"${random.randint(1, 5000)}."


tools = [get_top_gainers_and_losers, forecast_stock_price]

text_model_with_tools = text_model.bind_tools(tools)

sys_msg = SystemMessage(
    content="""You are an investment analyst equipped with tools such as stock price forecasting and getting top gainers and losers in the market. 
If using tools is not relevant to the user's question, just return a general answer without using tools."""
)

def agent(state: MessagesState):
    return {"messages": [text_model_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
)
builder.add_edge("tools", "agent")
graph = builder.compile()
```

- let's run the graph with the same inputs we've tried before:

```python
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="What are the top gainers and losers in the market?")]},
    config,
)
print(result)

config2 = {"configurable": {"thread_id": "2"}}
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What is the forecasted stock price for Apple?")]},
    config2,
)
print(result2)

config3 = {"configurable": {"thread_id": "3"}}
result3 = graph.invoke(
    {"messages": [HumanMessage(content="hello")]},
    config3,
)
print(result3)
```

- now, let's run a conversation within the same thread, to illustrate how the ReAct pattern allows for dynamic tool usage and a more natural conversation flow, for this we are going to slightly modify the tools:

```python
@tool
def get_top_gainers_and_losers() -> str:
    """Get today's top gainers and losers in the market.

    Use this tool to get the top gainers and losers in the market.
    """
    return f"Apple lost 25% of its value today as Tim Cook died. It is now valued at $150."

@tool
def forecast_stock_price(initial_price: int) -> str:
    """Forecast the stock price of a given company.

    Use this tool to forecast the stock price of a given company.
    """
    return f"${initial_price + random.randint(1, 5000)}."

result = graph.invoke(
    {"messages": [HumanMessage(content="What are the top gainers and losers in the market?")]},
)
print(result)

result2 = graph.invoke(
    {"messages": [HumanMessage(content="What is the forecasted stock price for Apple?")]},
)
print(result2)

result3 = graph.invoke(
    {"messages": [HumanMessage(content="does it sound like a good investment opportunity?")]},
)
print(result3)
```

- the ReAct pattern is adapted to simple single-objective tasks that are to be triggered by the user in real-time, it is to be used in quick-response scenarios, for example in a chatbot