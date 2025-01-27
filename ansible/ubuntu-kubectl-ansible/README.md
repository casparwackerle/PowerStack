# Ubuntu Kubectl Ansible

This directory contains Ansible playbooks and roles to deploy an **Ubuntu pod with kubectl** inside a Kubernetes cluster. The pod acts as a **benchmarking environment**, allowing various resource-intensive workloads to run within a controlled Kubernetes namespace.

## Overview

The automation performs the following tasks:
- Deploys an **Ubuntu container** with `kubectl` pre-installed.
- Sets up **high-CPU and high-memory workloads** in separate namespaces.
- Installs **benchmarking tools** such as:
  - `stress-ng` (for CPU stress tests)
  - `fio` (for disk I/O benchmarking)
  - `iperf3` (for network performance tests)
  - `htop` (for system monitoring)
- Configures a **dedicated disk** for benchmarking.

## Playbooks

### `install-ubuntu-kubectl.yml`
Deploys an Ubuntu pod with:
- `kubectl`
- Benchmarking tools
- Preconfigured Kubernetes authentication.

### `remove-ubuntu-kubectl.yml`
Removes the Ubuntu benchmarking pod and associated configurations.

## Roles

### `deploy-benchmarking-pod`
- Deploys an **Ubuntu-based Kubernetes pod** for running benchmarks.
- Ensures **kubeconfig** is accessible inside the pod.

### `deploy-pods-high-cpu`
- Creates multiple **high-CPU workload deployments**.
- Labels the control node for **affinity-based scheduling**.

### `deploy-pods-high-mem`
- Deploys high-memory-consuming pods.
- Defines resource limits and requests.

### `install-tools`
- Installs various **benchmarking tools** inside the Ubuntu container.

### `setup-openssh`
- Ensures SSH connectivity inside the pod for remote testing.

### `install-kubecontrol`
- Configures **kubectl** inside the Ubuntu pod.
- Ensures Kubernetes authentication is properly set up.

### `post-install-cleanup`
- Cleans up temporary files and manifests after deployment.