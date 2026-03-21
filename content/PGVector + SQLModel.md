---
tags:
  - databases
  - python
  - swe
  - llm
  - fastapi
folder: learning
share: true
title: PGVector + SQLModel
date created: Sunday, March 3rd 2024, 6:05:15 pm
date modified: Thursday, March 7th 2024, 11:25:18 pm
---

## create the embedding and write to the database manually

Adapt the SQLModel [[./populate a PostgreSQL database with SQLModel|model]] (see [[./word2vec|word2vec]]) with a [column for the embedding](https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlmodel)

```python
class Player(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	name: str
	foot: str
	club_id: Optional[int] = Field(default=None, foreign_key="team.id")
	club: Optional[Club] = Relationship(back_populates="players")
	quote: Optional[str] = None
	embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(384))) # change embedding size according to model used
```

Then, use a [sentence transformer model](https://huggingface.co/sentence-transformers) to embed the document

```python
ronaldo_embedding = embeddings_model.encode("Siuuuuu")
ronaldo = Player(
	name="Cristiano Ronaldo",
	foot="right",
	club="Real Madrid"
	quote="Siuuuuu"
	embedding=ronaldo_embedding
)
```

and then [[./populate a PostgreSQL database with SQLModel|write to the database]]. Might have to run `session.exec(text("CREATE EXTENSION IF NOT EXISTS vector))` first.

## writing to the database using LangChain

Or use [LangChain to write](https://python.langchain.com/docs/integrations/vectorstores/pgvector#similarity-search-with-euclidean-distance-default) (with less flexible column and table names).

```python
PGVector.from_documents(
	documents=documents, # need to be List[Documents]
	embedding=embeddings_model,
	ids=[documents[i].metadata["player_id"] for i in range(len(documents))], # create `custom_id` column
	connection_string=db_url,
)
```

We can then create a session without having to rewrite to the database using `from_existing index`

```python
def get_vector_db():
	db = PGVector.from_existing_index(
		embedding=embeddings_model,
		connection_string=db_url,
	)
	yield db
```

Session management can be used in a FastAPI function with

```python
def vector_db_func(
	vector_db: VectorStore = Depends(get_vector_db)
):
	pass
```
