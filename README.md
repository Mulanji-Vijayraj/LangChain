# LangChain - Short Notes

LangChain is a modular framework that simplifies the development of applications using large language models (LLMs). It enables chaining components like prompts, models, tools, and memory to build robust LLM pipelines for agents, retrieval systems, and more.

---

## Overview

LangChain enables:

- Composable chains with LLMs and tools
- Prompt engineering and templating
- Memory integration (for context retention)
- Agent-based systems
- Retrieval-Augmented Generation (RAG)

---

## Packages and Key Modules

### `langchain.llms`

**Purpose:** Interface with LLM providers (OpenAI, HuggingFace, etc.)

```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
```

---

### `langchain.prompts`

**Purpose:** Create prompt templates for LLMs.

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
```

---

### `langchain.chains`

**Purpose:** Combine LLMs and prompts into executable chains.

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("toys"))
```

---

### `langchain.agents`

**Purpose:** Create LLM agents that use tools (e.g., calculator, search).

```python
from langchain.agents import load_tools, initialize_agent

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Who won the world series in 2023?")
```

---

### `langchain.memory`

**Purpose:** Maintain conversation history or context.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
```

---

### `langchain.vectorstores` + `langchain.embeddings`

**Purpose:** Use embeddings + vectorstores for retrieval-based QA (RAG).

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
```

---

## Example

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.5)
prompt = PromptTemplate(
    input_variables=["language"],
    template="Translate 'I love programming.' into {language}."
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("French"))
```

---

## Use Cases
- Conversational Agents
- Knowledge Base QA (RAG)
- Text summarization
- Code generation and debugging
- AI tutors

---

## Installation

1. Download the requirements.txt file from the repository.
2. In the terminal, notebook, etc. run the command:

```python
pip install -r requirements.txt
```

3. And start coding with LangChain! (The process is similar for langGraph and LangSmith too)

---

# Thank you for reading!
