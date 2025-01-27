# Monitoring Stack Deployment

This directory contains Ansible playbooks and roles for deploying and managing the monitoring stack in a Kubernetes environment.

## Playbooks
- **`deploy-monitoring-stack.yml`** – Installs and configures the monitoring stack using Helm.
- **`remove-monitoring-stack.yml`** – Uninstalls the monitoring stack and cleans up related resources.

## Deployment Overview
1. Ensures required **namespaces** and **storage directories** exist.
2. Configures **persistent storage** for monitoring components (Grafana, Prometheus, AlertManager).
3. Deploys the **Kube-Prometheus Stack** using Helm.
4. Configures **port-forwarding** for Grafana, Prometheus, and AlertManager.
5. Performs post-install cleanup.

## Roles
- **`persistent-storage`** – Ensures persistent storage is correctly configured.
- **`install-monitoring-stack`** – Handles the deployment of monitoring components.
- **`port-forward-grafana`** – Sets up Grafana access.
- **`port-forward-prometheus`** – Enables Prometheus access.
- **`port-forward-alertmanager`** – Configures AlertManager port-forwarding.
- **`post-install-cleanup`** – Cleans up temporary files and resources.

## Notes
- Requires **Helm** for deployment.
- Uses **NFS storage** for persistent monitoring data.
- Grafana is configured with a static NodePort for easier access.
- **Ensure proper firewall and access settings** to allow external connections.

---
For more details, refer to the [thesis documentation](../thesis/VT1/build/main.pdf).