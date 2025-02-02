# instrumenting LLMs basics

- prerequisites from `README.md`
- `ollama run llama3.1` to get a plain text response from text input
- call Ollama API to get a structured output from text input:

```bash
curl -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
  "model": "llama3.1",
  "messages": [{"role": "user", "content": "Tell me about Apple."}],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "cie_description": {
        "type": "string"
      },
      "cie_symbol": {
        "type": "string"
      },
      "sector": {
        "type": "string"
      },
      "industry": {
        "type": "string"
      }
    },
    "required": [
      "cie_description",
      "cie_symbol",
      "sector",
      "industry"
    ]
  }
}'
```

... this is more reliable and consistent than JSON mode in which you need to specify the schema directly in your prompt.
Structured outputs are very interesting for manipulating natural language at scale, for example: parsing data from documents, extracting data from images, structuring all language model responses, etc.

- `python3 -m venv venv && source venv/bin/activate`
- create a `requirements.txt` file with the following content:

```
langchain-core
langchain-ollama
ollama
pydantic
```

- `pip install -r requirements.txt`
- get plain text response from text input using LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Process this text: {input}")

llm = ChatOllama(model="llama3.1")

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "input": "Apple Inc. is a multinational technology company headquartered in Cupertino, "
            "California that designs, manufactures, and markets consumer electronics, "
            "computer software, and online services."
})
print(result)
```

- get structured output from text input

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")


model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(CompanyInfo)

prompt = ChatPromptTemplate.from_messages(
    [("system", "Extract information from this text:"), ("human", "{input}")]
)

chain = prompt | structured_model

result = chain.invoke(
    {
        "input": "Apple Inc. is a multinational technology company headquartered in Cupertino, "
        "California that designs, manufactures, and markets consumer electronics, "
        "computer software, and online services."
    }
)
print(result)
```

- `ollama pull minicpm-v`
- plain text response from image input

```python
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

vision_model = ChatOllama(
    model="llama3.2-vision"
)
with open("1-instrumenting-llms-basics/example.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = vision_model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
    ])
])

print(response)
```

- structured output from image input

```python
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")

vision_model = ChatOllama(
    model="llama3.2-vision"
)
structured_model = vision_model.with_structured_output(CompanyInfo)

with open("example.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = structured_model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
    ])
])

print(response)
```

... this fails because `llama3.2-vision` does not support "tools", which are required for structured outputs. This means we need to write a chain to call the vision model first and then structure the output with a text model; this allows use to consider a more elaborate chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
import base64


class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")


vision_model = ChatOllama(model="llama3.2-vision")
text_model = ChatOllama(model="llama3.1").with_structured_output(CompanyInfo)

vision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {"type": "text", "text": "Describe this image in detail"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image}"},
            ],
        )
    ]
)
analysis_prompt = ChatPromptTemplate.from_template(
    """
Extract corporate information from this description, taken from an image:
{description}

Provide details using this format:
{format_instructions}
                                                   
DO NOT reference the image in your response.                                                  
"""
).partial(format_instructions=CompanyInfo.model_json_schema())

vision_then_structured_chain = (
    vision_prompt
    | vision_model
    | (lambda x: {"description": x.content})
    | analysis_prompt
    | text_model
)

# Process image
with open("example.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

result = vision_then_structured_chain.invoke({"image": image_b64})
print(result)
```
