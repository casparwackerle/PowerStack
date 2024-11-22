# Installation

## General Information

This repository is designed for bare-metal nodes with very specific goals and requirements (see [README](../README.md)). While some components may work under different circumstances, the entire stack is unlikely to function in a drastically different environment. Additionally, please note that this project is not primarily intended for use by anyone besides its owner. A fully functional deployment is not guaranteed.

---

## Prerequisites

### Servers
- Two or more physical servers with Ubuntu 22.04 installed. This project was tested on x64 architecture.
- Passwordless SSH access to all servers.
- A user account with `sudo` privileges.
- The control node *must* have Ansible 8.0+ (ansible-core 2.15+).
- The control node must have an empty storage device available, which will be used to provide persistent storage for Kubernetes.
- Networking between the physical servers. A subnet for internal communication is recommended but not required.
- It is recommended that all managed nodes disable firewalls and swap. See [K3s Requirements](https://docs.k3s.io/installation/requirements) for more information.

### Software
- **Ansible**: Version 8.0+.
- **Kubectl**: Version v1.30.2+.
- **Helm**: Version v3.16.2+, initialized

---

## Usage

### Pre-Installation

#### Clone the Project
```bash
git clone --recurse-submodules -j8 git@github.com:casparwackerle/PowerStack.git
```
#### Ansible Inventory
```bash
cp ansible/configs/inventory_example.yml ansible/configs/inventory.yml
vi ansible/configs/inventory.yml
```
Edit the [Ansible Inventory File](../ansible/configs/inventory.yml) to match your desired configuration, specifically:
- Internal and external IP addresses of each node. These may be the same if you are not using an internal network.
- Ansible user for SSH server access.
- Rancher hostname, which will expose the Rancher Kubernetes management platform.
- NFS network, path, and disk to be used. It is assumed that the specified disk belongs to the control node to enable the dynamic shutoff of worker nodes.
- Size of Kubernetes PV and PVC, ensuring it fits within the NFS disk capacity.


**Disclaimer**: The selected disk for NFS will be reformatted, resulting in the loss of any existing data.

---

#### Configure the Ansible Vault
```bash
cp ansible/configs/vault-template.yml ansible/configs/vault.yml
vi ansible/configs/vault.yml
```
Edit the vault file to replace placeholder tokens and passwords with your own. You can generate secure tokens using:
```bash
pwgen -s 64 1
# OR
openssl rand -base64 64
```
After updating the tokens, encrypt the vault file:
```bash
ansible-vault encrypt ansible/configs/vault.yml
```

---

### Installation
Run the [Deploy All Script](../scripts/deploy_all.sh) to initiate the installation process:
```bash
. scripts/deploy_all.sh
```
**Disclaimer**: You will be prompted *multiple* times for the Ansible Vault password during the installation process.

**NOTE**: In the event of failure, logs can be found in the `/logs` directory.

---

### Post-installation
#### Set up `kubectl` access on your local machine
After the cluster is successfully deployed, the kubeconfig file will be copied to your local machine at `~/.kube/config.new`with the `powerstack` context. Assuming you have [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) installed, follow these steps:
1. Copy the kubeconfig file:
```bash
cp ~/.kube/config.new ~/.kube/config
```
2. Switch to the `powerstack` context:
```bash
kubectl config use-context powerstack
```
3. Verify cluster access:
```bash
kubectl get nodes -o wide
```
Ensure that kubectl is using the correct configuration file:
```bash
export KUBECONFIG=~/.kube/config
```
---

#### Ensure Access to Rancher
To access rancher, update your local DNS or `hosts` file:
```bash
echo "<control_node_expernal_IP> <rancher_hostname>" | sudo tee -a /etc/hosts
```
When accessing the Rancher interface for the first time, you will be asked for the *Bootstrap Password*, which you defined and encrypted in the Ansible Vault.

---

### Additional Notes
- This installation process assumes familiarity with basic Linux commands and networking.
- Use a testing environment to experiment before deploying on production servers.