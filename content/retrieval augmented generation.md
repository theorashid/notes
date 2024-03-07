---
tags:
  - llm
  - databases
  - python
  - sql
  - ml
folder: llm
share: true
title: retrieval augmented generation
date created: Thursday, March 7th 2024, 11:03:29 pm
date modified: Thursday, March 7th 2024, 11:23:59 pm
---

When giving a prompt to a LLM such as

```python
prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.

Question: {question}

Context: {context}

Answer:
```

providing the entire context from the database is very costly. Instead, we can use RAG to **provide only the most relevant context**.

Set up the embeddings in a [[./PGVector + SQLModel|vector database]] . Then, embed the query (`question`) and find the vector [similarity](https://www.pinecone.io/learn/vector-similarity/) (usually some sort of dot product) between the query and the embeddings to find the most relevant `context`.

The search index finds **approximate matches** rather than exact match in scalar indexing. See some different [index strategies](https://www.datastax.com/guides/what-is-a-vector-index).

We can use [LangChain to perform RAG](https://python.langchain.com/docs/use_cases/question_answering/sources).

```python
query = "Some sort of query?"

llm = ChatOpenAI()
retriever = vector_db.as_retriever(search_kwargs={"k": 6})

chain = (
	{"context": retriever, "question": RunnablePassthrough()}
	| prompt
	| llm
)

with get_openai_callback() as cb:
	result = chain.invoke(query)
	print(cb) # info on cost of the query
```
