---
tags:
  - swe
folder: learning
share: true
title: cross-account sts assume role
date created: Saturday, August 16th 2025, 11:36:15 am
date modified: Saturday, August 16th 2025, 11:43:25 am
---

Say we want a specific worker role to assume another data access role with the relevant permissions to the dataset.

## in the same aws account

Edit the trust relationship of the data access role in account A to give permission to the worker role in account A.

```
{
	"Effect": "Allow",
	"Principal": {
		"AWS": "arn:aws:iam::<account-a-id>:role/<worker-role>"
	},
	"Action": "sts:AssumeRole"
}
```

## cross-account

In the data access role is now in account B, as well as editing the trust relationship of the data access role, we also need to add a policy to the worker role in account A to say it *can* assume the data access role in account B.

```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": "sts:AssumeRole",
			"Resource": "arn:aws:iam::<account-b-id>:role/<data-access-role>"
		}
	]
}
```
