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
date modified: Friday, April 5th 2024, 1:52:55 pm
---

When giving a prompt to a LLM such as

```python
prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.

Question: {question}

Context: {context}

Answer:
```

providing the entire context from the database is very costly. Instead, we can use RAG to **provide only the most relevant context**.

Set up the embeddings in a [[./PGVector + SQLModel|vector database]]. Then, embed the query (`question`) and find the vector [similarity](https://www.pinecone.io/learn/vector-similarity/) (usually some sort of dot product) between the query and the embeddings to find the most relevant `context`.

The search index finds **approximate matches** rather than an exact match in scalar indexing. See some different [index strategies](https://www.datastax.com/guides/what-is-a-vector-index).

If we have manually set up the embeddings table, we can use [SQLModel](https://github.com/pgvector/pgvector-python#sqlmodel) to do the **retrieval** step to get the `k` most relevant documents.

```python
embedding_vector = embeddings_model.encode(query)
statement = select(Document).order_by(Document.embedding.cosine_distance(embedding_vector)).limit(k)
documents = session.exec(statement).all()
```

We can also use [LangChain to perform all RAG steps](https://python.langchain.com/docs/use_cases/question_answering/sources), taking advantage of the `vector_db.as_retriever` functionality.

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
