---
tags:
  - databases
  - python
  - swe
folder: learning
share: true
title: populate a PostgreSQL database with SQLModel
date created: Sunday, March 3rd 2024, 6:03:29 pm
date modified: Saturday, April 13th 2024, 11:40:54 am
---

Create models, which are effectively columns in the database (see the [SQLModel docs](https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/) for details; consider [[./sqlmodel (sqlalchemy) cascade|sqlmodel (sqlalchemy) cascade]] in `Relationship`).

```python
class Club(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	name: str
	players: List["Player"] = Relationship(back_populates="club")

class Player(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	name: str
	foot: str
	quotes: Optional[str] = None
	club_id: Optional[int] = Field(default=None, foreign_key="team.id")
	club: Optional[Club] = Relationship(back_populates="players")
```

Then, [[./maintaining privacy to connect to a PostgreSQL database with pydantic|connect to the database]] and write from JSON records format.

```python
sql_model_objs = [Model(**entry) for entry in json_records]

with Session(engine) as session:
	for sql_model_obj in sql_model_objects:
		Model.model_validate(sql_model_obj)
		session.add(sql_model_obj)
	# or session.bulk_save_objects(sql_model_objects)
	session.commit()
```

If writing from a pandas dataframe, make sure the `NaN` values are of the correct type. This can be forced with `df = df.astype({"quotes": "string"})`.

These can be used with [FastAPI](https://sqlmodel.tiangolo.com/tutorial/fastapi/) (with [tests](https://sqlmodel.tiangolo.com/tutorial/fastapi/tests/)).
