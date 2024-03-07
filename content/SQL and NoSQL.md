---
tags:
  - swe
  - databases
  - sql
folder: learning
title: SQL and NoSQL
date created: Saturday, February 17th 2024, 1:12:06 pm
date modified: Thursday, March 7th 2024, 10:08:13 pm
share: true
---

## SQL

Use PostgreSQL most of the time.

### querying

```SQL
SELECT c.id, column AS column_alias
FROM table_names
WHERE column = condition / IN / IS NOT NULL
GROUP BY
HAVING
ORDER BY
LIMIT n
```

### joining

explicit:

```SQL
LEFT/RIGHT/FULL INNER/OUTER JOIN
WHERE x.id = y.id
```

implicit:

```SQL
SELECT table1.column_name1, table2.column_name2, table3.column_name3
FROM table1, table2, table3
```

## NoSQL

- Probably don't need NoSQL scaling unless you have PB of data.
- Allows *horizontal* scaling (more boxes, rather than *vertical* (bigger boxes)) by creating schema-less **shards** of data as key-value store (dictionary, JSON).

**Cap theorem**. You can only have 2 out of 3 of:

- Consistency, all clients see the same view of data
- Availability, all clients can find a replica of data
- Partitioning, system works in the presence of a partial network failure

MongoDB has a main node (for consistency). Apache Cassandra has no main node (for availability).
