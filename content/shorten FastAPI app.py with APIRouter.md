---
tags:
  - fastapi
  - swe
  - python
folder: learning
share: true
title: shorten FastAPI app.py with APIRouter
date created: Thursday, March 7th 2024, 10:23:56 pm
date modified: Thursday, March 7th 2024, 10:28:47 pm
---

FastAPI `app.py` files can get long. Use `APIRouter` to [structure the application into different files](https://fastapi.tiangolo.com/tutorial/bigger-applications/). In `api/elsewhere.py`,

```python
router = APIRouter(prefix="/db")

@router.get("/somewhere/")
def func():
	pass
```

and then in `app.py`

```python
from .api import elsewhere

app = FastAPI()
app.include_router(elsewhere.router)
```
