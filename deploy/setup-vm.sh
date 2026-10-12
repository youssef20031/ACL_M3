#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 22.04/24.04 VM (x86_64 or arm64; e.g. an
# Azure B-series VM or Oracle Cloud Ampere). Run from the repo root on the VM:
#
#   bash deploy/setup-vm.sh
#
# Installs Docker, adds swap on small VMs, opens ports 80/443 in the host
# firewall, and creates deploy/.env with a generated Neo4j password and an
# sslip.io hostname.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking swap"
# A 4 GB VM runs the stack fine, but the image build and model loading can spike;
# swap turns a would-be out-of-memory kill into a brief slowdown.
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if [ "$(swapon --noheadings | wc -l)" -eq 0 ] && [ "$mem_kb" -lt 16000000 ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo "    Added a 4 GB swap file"
else
  echo "    Swap already present or plenty of RAM; skipping"
fi

echo "==> Installing Docker Engine + compose plugin (official apt repository)"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "    Docker installed. You'll need to log out and back in (or run 'newgrp docker') before using docker without sudo."
else
  echo "    Docker already installed"
fi

echo "==> Opening ports 80 and 443 in the host firewall"
# Needed on Oracle's Ubuntu images, whose iptables rules REJECT everything except
# SSH; harmless elsewhere. The cloud-level firewall (Azure Network Security Group,
# Oracle Security List) is opened separately in the provider's console.
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT
  fi
done
if ! sudo iptables -C INPUT -p udp --dport 443 -j ACCEPT 2>/dev/null; then
  sudo iptables -I INPUT -p udp --dport 443 -j ACCEPT   # HTTP/3
fi
if command -v netfilter-persistent >/dev/null 2>&1; then
  sudo netfilter-persistent save
else
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save
fi

echo "==> Creating deploy/.env"
if [ -f deploy/.env ]; then
  echo "    deploy/.env already exists; leaving it alone"
else
  cp deploy/.env.example deploy/.env
  public_ip="$(curl -fsS https://api.ipify.org || true)"
  if [ -n "$public_ip" ]; then
    sed -i "s/^SITE_ADDRESS=.*/SITE_ADDRESS=${public_ip//./-}.sslip.io/" deploy/.env
  fi
  sed -i "s/^NEO4J_PASSWORD=.*/NEO4J_PASSWORD=$(openssl rand -hex 24)/" deploy/.env
  chmod 600 deploy/.env
  echo "    Wrote deploy/.env (SITE_ADDRESS=${public_ip//./-}.sslip.io)."
  echo "    Now add your GROQ_API_KEY to deploy/.env."
fi

echo
echo "Next steps:"
echo "  1. Edit deploy/.env and set GROQ_API_KEY"
echo "  2. bash deploy/deploy.sh            # build and start everything"
echo "  3. bash deploy/deploy.sh --load-data # first time only: load the graph"
echo "  4. bash deploy/install-daily-job.sh   # daily current-season refresh"
