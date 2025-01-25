---
tags:
  - swe
folder: learning
share: true
title: kubernetes
date created: Monday, January 13th 2025, 7:52:48 pm
date modified: Monday, January 13th 2025, 10:07:11 pm
---

Kubernetes Cluster:

- **[Nodes](https://kubernetes.io/docs/concepts/overview/components/#node-components) (also called workers**) are the physical or virtual machines that make up the Kubernetes cluster. These are provisioned in [[./infrastructure as code|CDK]] with `cluster.addNodeCapacity()`.
    - runs the [container](https://kubernetes.io/docs/concepts/containers/) runtime (Docker)
    - runs **kubelet** (agent that communicates with the control plane)
    - runs kube-proxy (network proxy)
- **Pods** are the smallest deployable unit. Single instance of a running process in your cluster. These are created using `kubectl apply` (`kubectl` does *not* create nodes, but it can create other custom resources like Ray.)
    - pods **run on nodes** (a Ray cluster "head-node" is a pod)
    - pods can contain one or more containers, which share the same network namespace
    - pods are ephemeral (created, destroyed, and rescheduled based on the cluster's needs)
    - pods are scheduled onto nodes by the Kubernetes scheduler
- **[Namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)** are **cluster-wide** resources. `kubectl get namespace`
    - do not directly affect or partition nodes. All namespaces can potentially use any node in the cluster
    - Pods must be created within a namespace. Pods from different namespaces can coexist on the same node.
    - for organising and isolating resources in multi-tenant environments
    - `'default'`, `'kube-system'`, `'kube-ray'`
- scaling
    - horizontal pod autoscaling: increasing or decreasing the number of pod replicas
    - cluster autoscaling: adding or removing *nodes* based on resource demands
- blueprints/templates/**manifests** are yaml/json configuration files that describe how a Kubernetes application and its associated resources should be deployed and managed
- **Helm** charts are packages of pre-configured Kubernetes resources (package management)
- [Control Plane](https://kubernetes.io/docs/concepts/overview/components/#control-plane-components) are components that manage the cluster: kube-api-server, kube-scheduler, kube-controller-manager, etcd (the cluster's database, key value store for all API server data)
