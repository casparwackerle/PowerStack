# Installation

## General Information

This repository is designed for bare-metal nodes with very specific goals and requirements (see [README](../README.md)). While some components may work under different circumstances, the entire stack is unlikely to function in a drastically different environment. Finally, please be aware that this project is not primarily inteded to be used by anyone besides its owner. A working project is not guaranteed.

## Prerequisites

### Servers
- Two or more physical servers with Ubuntu 22.04 installed.
- Passwordless SSH access to all servers.
- User account with `sudo` privileges.
- The control node *must* have Ansible 8.0+ (ansible-core 2.15+)
- The control node must have an empty storage device available, which will be used to provide persistent storage for Kubernetes.
- Networking between the physical servers. A subnet for internal communication is recommended but not required.
- It is recommended that all managed nodes disable firewalls and swap. See [K3s Requirements](https://docs.k3s.io/installation/requirements) for more information.

### Software


## Usage

### Project setup
Clone the project including its submodules
```bash
git clone --recurse-submodules -j8 git@github.com:casparwackerle/PowerStack.git
```
In the [Ansible Inventory File](../ansible/configs/inventory.yml), make the necessary changes to match your desired configuration, most importantly:
- Internal and external IP address of each node. These may be the same if you are not using an internal network.
- ansible user for SSH server access
- Rancher hostname, which will expose the rancher
- NFS network, path and disk to be used. It is assumed that the used disk is part of the control node to allow the dynamic shutoff of worker nodes.
- Size of kubernetes PV and PVC, must fit on the NFS disk

**DISCLAIMER**: The selected NFS disk will be reformatted, resulting in the loss of any data on it. 