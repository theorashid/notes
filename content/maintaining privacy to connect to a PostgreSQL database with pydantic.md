---
tags:
  - databases
  - python
  - swe
  - fastapi
folder: learning
share: true
title: maintaining privacy to connect to a PostgreSQL database with pydantic
date created: Sunday, March 3rd 2024, 6:04:28 pm
date modified: Thursday, March 7th 2024, 11:25:40 pm
---

We do not want to upload the real password a username to a public repository. Instead, set up a local `.env` with

```
USER=<username>
PASSWORD=<password>
DATABASE_NAME=<db>
PORT="5432"
```

This `.env` is in the `.gitignore` so is never committed to the public repository. Then, create a `config.py` file with a [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/#dotenv-env-support) class.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
	user: str
	password: str
	database_name: str
	port: str = "5432"

	model_config = SettingsConfigDict(
		env_file=find_dotenv(usecwd=True), # default behaviour is only to search cwd
		env_file_encoding="utf-8", # handle special characters in passwords
		extra="ignore",
	)
```

We can then call these settings to connect to the database.

```python
@lru_cache
def get_db_settings():
	return DatabaseSettings()

conf = get_db_settings()

db_url = f"postgresql://{conf.user}:{conf.password}@localhost:{conf.port}/{conf.database_name}"
engine = create_engine(db_url)

def init_db():
	SQLModel.metadata.create_all(engine)

def get_session():
	with Session(engine) as session:
		yield session
```

Using `lru_cache` means these variables are [only read once from disk](https://fastapi.tiangolo.com/advanced/settings/#creating-the-settings-only-once-with-lru_cache).

If the variables are required in a pipeline (for example, during testing), then these environment variables in the `.env` must be added as CI/CD variables in pipeline settings.
