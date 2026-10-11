# Deploying to a single VM (Oracle Cloud Free Tier)

Everything runs on one VM with Docker Compose:

| Service | What it does |
|---|---|
| `web` | Caddy: automatic HTTPS, serves the built React app, proxies `/api`, `/health`, `/ws` to the backend |
| `backend` | FastAPI app (`api_main.py`) |
| `neo4j` | Neo4j 5.26 Community, data in a Docker volume (not exposed publicly) |

No Railway, Vercel or Aura needed. The graph can always be rebuilt from the CSVs in the repo, so the only thing worth backing up is `deploy/.env`.

## 1. Create the VM (Oracle Cloud console)

1. **Compute → Instances → Create instance**
   - Image: **Canonical Ubuntu 24.04** (or 22.04), *aarch64*.
   - Shape: **Ampere → VM.Standard.A1.Flex**. 2 OCPU / 12 GB RAM is plenty; the Always Free allowance is 4 OCPU / 24 GB in total.
   - Networking: a public subnet, with **Assign a public IPv4 address** on.
   - Add your SSH public key.
   - Boot volume: 50 GB or more (Always Free includes 200 GB in total).
2. If you get **"Out of capacity"**, try another availability domain or retry later. This is common for free Ampere shapes.
3. Open the firewall at the cloud level: **Networking → Virtual cloud networks → your VCN → Security Lists → Default → Add ingress rules**
   - Source `0.0.0.0/0`, TCP, destination port `80`
   - Source `0.0.0.0/0`, TCP, destination port `443`
   - (optional) Source `0.0.0.0/0`, UDP, destination port `443` for HTTP/3

> **Idle reclamation:** Oracle may reclaim *Always Free* instances that stay mostly idle for 7 days on accounts that haven't been upgraded. Upgrading the account to **Pay As You Go** stops this and also makes Ampere capacity easier to get. Always Free resources stay free after the upgrade, but set a budget alert to be safe.

## 2. Set up the VM

```bash
ssh ubuntu@<vm-public-ip>
git clone https://github.com/youssef20031/ACL_M3.git
cd ACL_M3
bash deploy/setup-vm.sh
```

If the repository is private, clone it with a read-only **deploy key** (GitHub → repo Settings → Deploy keys) or a fine-grained personal access token.

`setup-vm.sh`:
- installs Docker
- opens ports 80/443 in the VM's own iptables (Oracle's Ubuntu images block them by default, in addition to the Security List)
- writes `deploy/.env` with a random Neo4j password and `SITE_ADDRESS=<ip-with-dashes>.sslip.io`

Log out and back in once so your user can run `docker` without `sudo`.

Then put your Groq key in `deploy/.env`:

```bash
nano deploy/.env        # set GROQ_API_KEY=...
```

## 3. Start it

```bash
bash deploy/deploy.sh --load-data   # first time: build, start, load the graph (~5-10 min on ARM)
```

Open `https://<SITE_ADDRESS>` (shown in `deploy/.env`). Caddy gets the certificate on the first request.

## Day-to-day

| Task | Command |
|---|---|
| Deploy the latest `main` | `bash deploy/deploy.sh` |
| Reload the graph after changing the CSVs or the loader | `bash deploy/deploy.sh --load-data` |
| Tail backend logs | `docker compose -f deploy/docker-compose.yml logs -f backend` |
| Status | `docker compose -f deploy/docker-compose.yml ps` |
| Neo4j Browser | `ssh -L 7474:localhost:7474 -L 7687:localhost:7687 ubuntu@<vm-ip>`, then open http://localhost:7474 (user `neo4j`, password from `deploy/.env`) |

**Automatic deploys:** `.github/workflows/deploy.yml` runs `deploy.sh` over SSH on every push to `main` once you add these repository secrets:
- `VM_HOST`
- `VM_USER` (`ubuntu`)
- `VM_SSH_KEY` (the private key of a pair whose public key is in the VM's `~/.ssh/authorized_keys`)
- `VM_APP_DIR` (optional)

Until then the workflow skips itself.

**Using your own domain later:** point an `A` record at the VM's IP, set `SITE_ADDRESS=your.domain` in `deploy/.env`, and run `bash deploy/deploy.sh`.

## Retiring the old hosting

Once the VM works, shut these down in their own dashboards:
- the Railway service (`aclm3-production`)
- the Vercel project
- optionally, the Aura instance

The repo no longer contains their configs.
