# Kepler Ansible Deployment

This directory contains Ansible playbooks and roles for automating the deployment and removal of **KEPLER** on a Kubernetes cluster.

## Playbooks
- **`deploy-kepler.yml`** – Installs and configures KEPLER on the cluster.
- **`remove-kepler.yml`** – Uninstalls KEPLER and cleans up resources.

## Deployment Overview
1. Ensures **perf\_event\_paranoid** is correctly configured on all nodes.
2. Installs **Helm** (if not already installed) and adds the KEPLER Helm repository.
3. Creates the KEPLER namespace (if missing) and sets up required directories.
4. Renders the KEPLER Helm values file and transfers it to the target machine.
5. Installs KEPLER using Helm, ensuring proper configuration.
6. Marks the installation as completed to prevent redundant deployments.

## Removal Process
The **removal playbook** ensures that KEPLER is completely uninstalled from the cluster, including:
- Deleting Helm releases
- Removing namespaces and associated resources
- Cleaning up configuration files and directories

## Roles
- **`perf-event-config`** – Configures performance monitoring settings on nodes.
- **`deploy-kepler`** – Handles the KEPLER installation and configuration.
- **`post-install-cleanup`** – Ensures unnecessary files and temporary resources are removed.

## Notes
- Requires **Ansible Vault** for securely managing sensitive configurations.
- Deployment assumes a Kubernetes cluster is already set up.
- Uses **Helm** for KEPLER installation and configuration.

---
For more details on KEPLER's role in PowerStack, refer to the [thesis documentation](../thesis/VT1/build/main.pdf).
