# tycho Ansible Deployment

This directory contains Ansible playbooks and roles for automating the deployment and removal of **tycho** on a Kubernetes cluster.

## Playbooks
- **`deploy-tycho.yml`** – Installs and configures tycho on the cluster.
- **`remove-tycho.yml`** – Uninstalls tycho and cleans up resources.

## Deployment Overview
1. Ensures **perf\_event\_paranoid** is correctly configured on all nodes.
2. Installs **Helm** (if not already installed) and adds the tycho Helm repository.
3. Creates the tycho namespace (if missing) and sets up required directories.
4. Renders the tycho Helm values file and transfers it to the target machine.
5. Installs tycho using Helm, ensuring proper configuration.
6. Marks the installation as completed to prevent redundant deployments.

## Removal Process
The **removal playbook** ensures that tycho is completely uninstalled from the cluster, including:
- Deleting Helm releases
- Removing namespaces and associated resources
- Cleaning up configuration files and directories

## Roles
- **`perf-event-config`** – Configures performance monitoring settings on nodes.
- **`deploy-tycho`** – Handles the tycho installation and configuration.
- **`post-install-cleanup`** – Ensures unnecessary files and temporary resources are removed.

## Notes
- Requires **Ansible Vault** for securely managing sensitive configurations.
- Deployment assumes a Kubernetes cluster is already set up.
- Uses **Helm** for tycho installation and configuration.

---
For more details on tycho's role in PowerStack, refer to the [thesis documentation](../thesis/VT1/build/main.pdf).
