#!/bin/bash

# Navigate to the rancher-ansible directory
cd ~/Documents/PowerStack/ansible/rancher-ansible || {
  echo "Directory ~/Documents/PowerStack/ansible/rancher-ansible does not exist."
  exit 1
}

# Run the Ansible playbook with the inventory file and ask for vault password
ansible-playbook playbooks/rancher.yml -i inventory.yml -i ../configs/inventory.yml --ask-vault-pass
