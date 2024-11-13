#!/bin/bash

# Navigate to the k3s-ansible directory
cd ~/Documents/PowerStack/ansible/k3s-ansible || {
  echo "Directory ~/Documents/PowerStack/ansible/k3s-ansible does not exist."
  exit 1
}

# Run the Ansible playbook with the inventory file and ask for vault password
ansible-playbook playbooks/site.yml -i inventory.yml -i ../configs/inventory.yml --ask-vault-pass
