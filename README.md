# PowerStack

## Description
**Disclaimer**: This project is currently in active development.

Here, a description of Powerstack is missing.

## Features
- Automated setup of a Kubernetes cluster using Ansible:
    - K3s
    - Rancher
    - NFS for a kubernetes Persistent Storage Volume (PV)
- Feature 2
- Feature 3

## Quick Start
1. Clone the repository, include recurse submodules.
2. Change the variables in the [ansible inventory](ansible/configs/inventory.yml) to suit your infrastructure.
3. Copy the [vault template](ansible/configs/vault-template.yml) to vault.yml in the same directory. Change the tokens. Encrypt the vault using ansible-vault.
4. Run [deploy_all.sh](scripts/deploy_all.sh)

**Disclaimer**: This deployment script will reformat a disk, causing potential data loss. Please read the installation guide.
**Disclaimer**: While all passwords are encrypted, the scripts in this repository cannot be considered high-security.

## Documentation
- [Project Documentation and Architecture](docs/project.md)
- [Installation Guide](docs/installation-and-setup.md)

## License
**Disclaimer**: This project includes third-party code with their respective licenses. It is the responsibility of users to ensure compliance with the licenses of included components. Once this project is done, I intend to clarify the License situation.
