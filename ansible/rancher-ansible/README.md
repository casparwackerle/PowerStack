# Rancher Ansible

This directory contains Ansible playbooks and roles for deploying **Rancher** on a Kubernetes cluster using Helm.

## Overview

This automation:
- Installs **Helm** (if not already installed).
- Adds the **Rancher Helm repository** and updates it.
- Installs **cert-manager** to handle TLS certificates.
- Deploys **Rancher** using Helm in the `cattle-system` namespace.
- Configures authentication and access.

## Playbooks

### `install-rancher.yml`
Deploys Rancher by:
1. Ensuring **Helm** is installed.
2. Adding the **Rancher Helm repository**.
3. Installing **cert-manager** (if missing).
4. Installing **Rancher** via Helm.
5. Setting up **TLS and authentication**.

## Roles

### `install-rancher`
- Installs Helm and updates repositories.
- Deploys Rancher in the `cattle-system` namespace.
- Configures authentication and optional **NodePort** access.
- Ensures cert-manager is installed for managing TLS certificates.
- Waits for cert-manager to be fully deployed.

## Grafana Dashboard

This directory also includes a **Grafana dashboard template** for KEPLER monitoring:
[KEPLER Grafana Dashboard](/ansible/rancher-ansible/Grafana-dashboard.JSON)
