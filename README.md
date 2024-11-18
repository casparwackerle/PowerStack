# PowerStack

## Description
**Disclaimer**: This project is currently in active development.

Here, a description of Powerstack is missing.

## Features
- Automated setup of a Kubernetes cluster using Ansible:
    - k3s
    - Rancher
    - NFS for a kubernetes Persistent Storage Volume (PV)
- Feature 2
- Feature 3

## Quick Start
1. Clone the repository, include recurse submodules.
2. Change the variables in the [ansible inventory](ansible/configs/inventory.yml) to suit your infrastructure.
3. Copy the [vault template](ansible/configs/vault-template.yml) to vault.yml in the same directory. Change the tokens. Encrypt the vault using ansible-vault.
4. Run [deploy_all.sh](scripts/deploy_all.sh)

## Documentation
- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)

## License
**Disclaimer**: This project includes third-party code with their respective licenses. It is the responsibility of users to ensure compliance with the licenses of included components. I will update this once I figure out which license to use myself.
