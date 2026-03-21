---
tags:
  - python
  - databases
  - swe
folder: learning
share: true
title: sqlmodel (sqlalchemy) cascade
date created: Saturday, April 13th 2024, 11:14:18 am
date modified: Saturday, April 13th 2024, 11:39:59 am
---

The default behaviour when a parent object is deleted is to make the child object orphans. For related objects to delete if they are not attached to their parent, we can use the `cascade` property in the parent class.

```python
class Club(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	name: str
	players: List["Player"] = Relationship(
		back_populates="club",
		sa_relationship_kwargs={"cascade": "delete"},
	)
```
