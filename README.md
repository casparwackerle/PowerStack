# PowerStack

## Description
**Disclaimer**: This project is currently in active development.

PowerStack is a Kubernetes-based infrastructure automation project designed for energy-efficient cloud computing. The system deploys a lightweight Kubernetes cluster using Ansible, with integrated monitoring and benchmarking tools to analyze energy consumption at the node and container level. PowerStack is primarily used for research on energy-efficient cloud systems and infrastructure optimization.

## Features
- **Automated Kubernetes Cluster Deployment:**
  - Lightweight K3s Kubernetes cluster
  - Rancher for cluster management
  - NFS-based Persistent Volumes (PV) for storage
- **Energy Efficiency Monitoring:**
  - KEPLER for real-time power consumption metrics
  - Prometheus for data collection
  - Grafana for visualization
- **Benchmarking and Stress Testing:**
  - Automated CPU, Memory, Disk, and Network I/O stress tests
  - Data collection and visualization for energy analysis
- **Infrastructure as Code:**
  - Fully automated Ansible playbooks for repeatable cluster setup
  - Secure Ansible Vault integration for sensitive data management

## Quick Start
> ⚠ **DISCLAIMER:** This process will reformat several disks and may result in **data loss**. Proceed with caution.
1. Clone the repository, including submodules:
   ```bash
   git clone --recurse-submodules https://github.com/your-repo/PowerStack.git
   ```
2. Update the Ansible inventory file at [`configs/inventory.yml`](configs/inventory.yml) to match your infrastructure.
3. Copy the vault template:
   ```bash
   cp configs/vault-template.yml configs/vault.yml
   ```
   - Modify the necessary tokens and credentials.
   - Encrypt the vault:
   ```bash
   ansible-vault encrypt configs/vault.yml
   ```
4. Run the deployment script:
   ```bash
   ./scripts/deploy_all.sh
   ```

**Disclaimer:**
While all passwords and credentials are encrypted, **this repository should not be considered a high-security environment**.

## Documentation
**Disclaimer:** The most detailed documentation of this project is the accompanying University thesis. All other documentation in the form of ReadMe files is kept short.
- The primary documentation for this is done as part of a university thesis. The most up-to-date version is here:
[Powerstack: Implementation of n energy monitoring environmnet in Kubernetes](thesis/VT1/build/main.pdf)
- An short installation guide is provided here:
- [Installation Guide](docs/installation-and-setup.md)
- [Architecture and Design](docs/architecture-and-design.md)

## License
**Disclaimer:** This project includes third-party software with their respective licenses. Users must ensure compliance with these licenses. The final licensing structure of PowerStack will be clarified once development stabilizes.
