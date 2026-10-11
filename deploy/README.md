# Deploying to a single VM

Everything runs on one VM with Docker Compose:

| Service | What it does |
|---|---|
| `web` | Caddy: automatic HTTPS, serves the built React app, proxies `/api`, `/health`, `/ws` to the backend |
| `backend` | FastAPI app (`api_main.py`) |
| `neo4j` | Neo4j 5.26 Community, data in a Docker volume (not exposed publicly) |

No Railway, Vercel or Aura needed. The graph can always be rebuilt from the CSVs in the repo, so the only thing worth backing up is `deploy/.env`.

Any Ubuntu 22.04/24.04 VM with **4 GB RAM** or more works, on x86_64 or arm64. Below are steps for Azure for Students (no credit card) and Oracle Cloud.

## 1a. Create the VM on Azure (Azure for Students)

1. In the [Azure portal](https://portal.azure.com), search **Virtual machines** → **Create** → **Azure virtual machine**.
2. **Basics** tab:
   - Subscription: **Azure for Students**. Resource group: **Create new**, e.g. `fpl`.
   - Virtual machine name: e.g. `fpl-vm`.
   - Region: any nearby region. If you get a *RequestDisallowedByPolicy* error, the student subscription doesn't allow that region; pick another.
   - Availability options: **No infrastructure redundancy required**.
   - Security type: **Standard**.
   - Image: **Ubuntu Server 24.04 LTS – x64 Gen2**.
   - Size: **Standard_B2s** or **Standard_B2als_v2** (2 vCPU, 4 GiB). The size picker shows the monthly price. These run about $25–35/month, so the $100 credit lasts roughly 3 months. Don't pick a 1 GiB size; it can't run the ML models.
   - Authentication: **SSH public key**, username `azureuser`, key source **Generate new key pair** (or use your existing key).
   - Inbound port rules: **Allow selected ports** → **SSH (22), HTTP (80), HTTPS (443)**.
3. **Disks** tab: OS disk type **Standard SSD**; the default size is enough.
4. **Networking** tab: keep the defaults (a new public IP is created; Standard public IPs are static, so the address won't change).
5. **Management** tab: make sure **Auto-shutdown** is **off**, or the site goes down every night.
6. **Review + create** → **Create**. If you generated a key, **download the `.pem` file** when prompted; it can't be downloaded again.
7. Open the new VM and copy its **Public IP address**.

Optional, for a nicer address than `<ip>.sslip.io`:
- On the VM's page, click the public IP → **Configuration** → set a **DNS name label** (e.g. `fpl-assistant`), then **Save**.
- You get `fpl-assistant.<region>.cloudapp.azure.com`. Put that in `SITE_ADDRESS` in step 2.

Connect from your computer (PowerShell works):

```bash
ssh -i path/to/fpl-vm_key.pem azureuser@<public-ip>
```

If Windows complains the key's permissions are too open, run this once in PowerShell:

```powershell
icacls path\to\fpl-vm_key.pem /inheritance:r /grant:r "$($env:USERNAME):R"
```

> **Credit:** when the student credit runs out, Azure disables the resources instead of charging you. Keep an eye on the balance at **Cost Management + Billing**, and plan a move (e.g. to Oracle or your own machine) before it runs low; the setup is the same anywhere.

## 1b. Or create the VM on Oracle Cloud (needs a credit or non-PIN debit card)

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

On Oracle the SSH user is `ubuntu` instead of `azureuser`, and with 12 GB+ RAM you can raise `NEO4J_HEAP=2G` and `NEO4J_PAGECACHE=1G` in `deploy/.env`.

## 2. Set up the VM

```bash
ssh azureuser@<vm-public-ip>     # ubuntu@ on Oracle
git clone https://github.com/youssef20031/ACL_M3.git
cd ACL_M3
bash deploy/setup-vm.sh
```

If the repository is private, clone it with a read-only **deploy key** (GitHub → repo Settings → Deploy keys) or a fine-grained personal access token.

`setup-vm.sh`:
- installs Docker
- adds a 4 GB swap file on small VMs
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
| Neo4j Browser | `ssh -L 7474:localhost:7474 -L 7687:localhost:7687 azureuser@<vm-ip>` (`ubuntu@` on Oracle), then open http://localhost:7474 (user `neo4j`, password from `deploy/.env`) |

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
