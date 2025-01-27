# NFS Ansible

This directory contains Ansible playbooks and roles for setting up an **NFS-based persistent storage** solution in the Kubernetes cluster.

## Overview

The playbooks automate:
- **Formatting and mounting an SSD** on the master node for persistent storage.
- **Installing and configuring an NFS server** on the master node.
- **Installing and configuring NFS clients** on worker nodes for shared storage access.

## Playbooks

### `setup-nfs.yml`
Sets up the **NFS server and clients**:
1. Formats and mounts an SSD as a persistent storage volume on the server.
2. Configures and starts the NFS server.
3. Configures NFS clients to mount the shared storage.

### `remove-nfs.yml`
Removes the NFS configuration and unmounts the storage from clients.

## Roles

### `format_ssd`
- Formats the specified SSD with **BTRFS**.
- Mounts the disk at `/mnt/data`.
- Ensures persistence via `/etc/fstab`.

### `nfs_server`
- Installs and configures **NFS Kernel Server**.
- Exports the storage directory to the Kubernetes cluster network.
- Restarts NFS services and ensures they run on boot.

### `nfs_client`
- Installs **NFS client utilities** on worker nodes.
- Mounts the NFS share on each client.
- Ensures persistence via `/etc/fstab`.