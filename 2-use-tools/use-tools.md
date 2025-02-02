# use tools

- prerequisites from `README.md`
- `python3 -m venv venv && source venv/bin/activate`
- create a `requirements.txt` file with the following content:

```
langchain-core
langchain-ollama
ollama
pydantic
```

- `pip install -r requirements.txt`
- let's see what a tool call looks like:

```python
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

result = text_model_with_tools.invoke(
    "What are the top gainers and losers in the market?"
)
print(result)

result2 = text_model_with_tools.invoke("What is the forecasted stock price for Apple?")
print(result2)

# ! this gives an undesired result because binding tools like constraints the model to only use the tools
result3 = text_model_with_tools.invoke("hello")
print(result3)
```

...then, we are to use the messages of type `tool_calls` to call the tools:

```python
for tool_call in result2.tool_calls:
    selected_tool = {
        "get_top_gainers_and_losers": get_top_gainers_and_losers,
        "forecast_stock_price": forecast_stock_price,
    }[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print(tool_msg)
```

- now, let's play the full interaction, from getting the user input, to generating a tool call signature, to invoking the tool, and then to passing the result back to the model for final answer to the user:

```python
from langchain_core.messages import HumanMessage
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

messages = [HumanMessage("I need you to issue a forecast for AAPL")]

ai_msg = text_model_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {
        "get_top_gainers_and_losers": get_top_gainers_and_losers,
        "forecast_stock_price": forecast_stock_price,
    }[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

result = text_model_with_tools.invoke(messages)
print(result)
```
