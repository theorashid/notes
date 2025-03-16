---
tags:
  - python
  - swe
folder: learning
share: true
title: ray on sagemaker
date created: Thursday, March 13th 2025, 9:36:56 pm
date modified: Saturday, March 15th 2025, 5:02:06 pm
---

*Adapted from [AWS samples for Ray](https://github.com/aws-samples/aws-samples-for-ray/blob/main/sagemaker/distributed_xgboost/src/sagemaker_ray_helper.py).*

The idea is to assign host `algo-1` as the **head node** (master host) of the [Ray cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). All other nodes (`algo-2`, `algo-3` etc) will be worker nodes.

```python
import subprocess
import os
import time
import ray
import socket
import json
import sys

class RayHelper():
	def __init__(
		self,
		ray_port:str="9339",
		redis_pass:str="redis_password"
	):
		self.ray_port = ray_port
		self.redis_pass = redis_pass
		self.resource_config = self.get_resource_config()
		# master_host is algo-1
		self.master_host = self.resource_config["hosts"][0]
		self.n_hosts = len(self.resource_config["hosts"])

	@staticmethod
	def get_resource_config():
		return dict(
			current_host = os.environ.get("SM_CURRENT_HOST"),
			hosts = json.loads(os.environ.get("SM_HOSTS"))
		)

	def _get_ip_from_host(self):
		ip_wait_time = 200
		counter = 0
		ip = ""

		while counter < ip_wait_time and ip == "":
			try:
				ip = socket.gethostbyname(self.master_host)
				break
			except:
				counter += 1
				time.sleep(1)

		if counter == ip_wait_time and ip == "":
			raise Exception(
				"Exceeded max wait time of {}s for hostname resolution".format(ip_wait_time)
			)
		return ip 

	def start_ray(self):
		master_ip = self._get_ip_from_host()
		if self.resource_config["current_host"] == self.master_host:
			output = subprocess.run(
				[
					'ray',
					'start',
					'--head',
					'-vvv',
					'--port',
					self.ray_port,
					'--redis-password',
					self.redis_pass,
					'--include-dashboard',
					'false'
				],
				stdout=subprocess.PIPE,
			)
		print(output.stdout.decode("utf-8"))
		ray.init(address="auto", include_dashboard=False)
		self._wait_for_workers()
		print("All workers present and accounted for")

		print("Available resources:")
		print(ray.available_resources())
		
		print("\nCluster resources:")
		print(ray.cluster_resources())

	else:
		time.sleep(10)
		output = subprocess.run(
			[
				'ray',
				'start',
				f"--address={master_ip}:{self.ray_port}",
				'--redis-password',
				self.redis_pass,
				"--block"
			],
			stdout=subprocess.PIPE,
		)
		print(output.stdout.decode("utf-8"))
		sys.exit(0)


def _wait_for_workers(self, timeout=60):
	print(f"Waiting up to {timeout} seconds for {self.n_hosts} nodes to join")

	while len(ray.nodes()) < self.n_hosts:
		print(f"{len(ray.nodes())} nodes connected to cluster")
		time.sleep(5)
		timeout-=5
		if timeout==0:
			raise Exception("Max timeout for nodes to join exceeded")
```

Then, instead of `ray.init()`, run `ray_helper = RayHelper(); ray_helper.start_ray()`.

This can be used to [batch process model runs](https://forum.pyro.ai/t/batch-processing-numpyro-models-using-ray/8718) or another embarrassingly parallel process.
