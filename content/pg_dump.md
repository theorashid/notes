---
tags:
  - databases
folder: learning
share: true
title: pg_dump
date created: Friday, March 22nd 2024, 8:47:32 pm
date modified: Sunday, March 24th 2024, 3:58:20 pm
---

Sometimes, if your database is small enough, and you don't want to pay for a database server, you can make up backup of the database by storing an archive file using [`pg_dump`](https://www.postgresql.org/docs/current/app-pgdump.html).

```sh
pg_dump db_name > dumpfile
```

Note that when restoring a database from file (`psql --set ON_ERROR_STOP=on db_name < dumpfile`), this will [reset the key sequence](https://stackoverflow.com/questions/4448340/postgresql-duplicate-key-violates-unique-constraint), and you get the error `duplicate key value violates unique constraint`. To avoid this error, the following function manually brings the sequence back in sync.

```sh
reset_sequence() {
	TABLES=$(sudo -u postgres psql -t -c "SELECT table_{name}FROM information_schema.columns WHERE column_{name='id'}AND table_schema='public'" $DATABASE_NAME)

	for table in $TABLES
	do
		echo "resetting sequence for table $table"
		PRIMARY_KEY="id"
		sudo -u postgres psql -c "SELECT setval(pg_get_serial_sequence('$table', '$PRIMARY_KEY'), coalesce(SELECT MAX(\"PRIMARY_KEY\") FROM "\"$table\"$, 0) + 1, false)" $DATABASE_NAME
	done
}
```
