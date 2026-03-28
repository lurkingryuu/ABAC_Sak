# Deploying Flask/FastAPI Apps on AWS Ubuntu 24.04

## Architecture

```
Internet
   │
   ▼
┌─────────────────────────────────────┐
│  Nginx Proxy Manager (Docker)       │
│  - Web UI for managing domains      │
│  - Auto SSL certs (Let's Encrypt)   │
│  Ports: 80, 443, 81 (admin)        │
└────────────┬────────────────────────┘
             │ forwards to 127.0.0.1:PORT
             │
     ┌───────┼───────────┐
     ▼       ▼           ▼
  App 1    App 2       App 3
  :5001    :5002       :5003
  (systemd + gunicorn services)
  (uv manages Python + deps)
```

**Key idea:**
- Each app runs as a **systemd service** with gunicorn, on a unique local port.
- **uv** handles Python versions and dependencies (via `pyproject.toml`). No manual venv fiddling.
- **Nginx Proxy Manager** (the only Docker thing) gives you a web UI to map domains → ports and auto-manage SSL.

---

## Part 1: Server Setup (One-Time)

SSH into your fresh Ubuntu 24.04 EC2 instance:

```bash
ssh -i your-key.pem ubuntu@YOUR_SERVER_IP
```

### 1.1 Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install essential tools

```bash
sudo apt install -y git curl
```

### 1.3 Install uv (Python package/project manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Verify:
```bash
uv --version
```

### 1.4 Add swap space (recommended on small/medium EC2 instances)

Many EC2 Ubuntu instances come without useful swap by default. Adding swap helps reduce OOM crashes during dependency installs, builds, or temporary traffic spikes.

Check current memory/swap first:

```bash
free -h
sudo swapon --show
```

Choose a swap size (practical starting point):

| RAM | Suggested Swap |
|-----|----------------|
| < 2 GB | 2x RAM |
| 2-8 GB | 1x RAM |
| 8-64 GB | 4-8 GB |
| > 64 GB | 4 GB (or workload-specific) |

Example below creates a 4 GB swap file:

```bash
# 1) Create swapfile (fast path)
sudo fallocate -l 4G /swapfile

# If fallocate is unsupported on your filesystem, use dd instead:
# sudo dd if=/dev/zero of=/swapfile bs=1M count=4096 status=progress

# 2) Lock down permissions
sudo chmod 600 /swapfile

# 3) Format + enable swap
sudo mkswap /swapfile
sudo swapon /swapfile

# 4) Verify
sudo swapon --show
free -h
```

Make swap persistent across reboot:

```bash
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
grep swapfile /etc/fstab
```

Optional tuning (good server defaults):

```bash
cat << 'EOF' | sudo tee /etc/sysctl.d/99-swap.conf
vm.swappiness=10
vm.vfs_cache_pressure=50
EOF

sudo sysctl --system
```

Notes:
- If `swapon /swapfile` returns "Invalid argument", recreate the file with `dd` (non-sparse) instead of `fallocate`.
- Swap is a safety net, not a RAM replacement. If swap is constantly high, increase instance memory.
- On EC2, swap on root EBS is common and persistent. If you place swap on instance-store volumes, it is lost when the instance stops/terminates.

### 1.5 Install Docker (for Nginx Proxy Manager only)

```bash
# Add Docker's official GPG key
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Let ubuntu user run docker without sudo
sudo usermod -aG docker ubuntu
```

**Log out and back in** for the group change to take effect:
```bash
exit
ssh -i your-key.pem ubuntu@YOUR_SERVER_IP
```

Verify:
```bash
docker --version
```

### 1.6 Create directory structure

```bash
mkdir -p ~/apps
mkdir -p ~/nginx-proxy-manager
```

### 1.7 Start Nginx Proxy Manager

```bash
cat > ~/nginx-proxy-manager/docker-compose.yml << 'EOF'
services:
  npm:
    image: jc21/nginx-proxy-manager:latest
    container_name: nginx-proxy-manager
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
      - "81:81"
    volumes:
      - ./data:/data
      - ./letsencrypt:/etc/letsencrypt
    extra_hosts:
      - "host.docker.internal:host-gateway"
EOF

cd ~/nginx-proxy-manager
docker compose up -d
```

Open `http://YOUR_SERVER_IP:81` in your browser:
- Email: `admin@example.com`
- Password: `changeme`
- **Change these immediately on first login.**

### 1.8 AWS Security Group

Make sure your EC2 security group allows inbound:

| Port | Purpose                     |
|------|-----------------------------|
| 22   | SSH                         |
| 80   | HTTP (needed for SSL certs) |
| 443  | HTTPS                       |
| 81   | NPM Admin (restrict to your IP!) |

---

## Part 2: Deploying an App

This is what you repeat for every new Flask/FastAPI app.

### 2.1 Requirements for every app

Your app repo needs:
- **`pyproject.toml`** — defines Python version and dependencies (uv uses this)
- **`app.py`** (Flask) or **`main.py`** (FastAPI) — your application entry point

That's it. No Dockerfile, no docker-compose, no virtualenv commands.

### 2.2 Clone and install

```bash
cd ~/apps
git clone https://github.com/youruser/your-app.git
cd your-app

# uv reads pyproject.toml, creates .venv, installs everything
uv sync
```

### 2.3 Set up environment variables

```bash
# Create .env file with your secrets
nano .env
```

Example:
```
RECAPTCHA_SITE_KEY=your-key-here
RECAPTCHA_SECRET_KEY=your-secret-here
```

### 2.4 Test it works

```bash
# Quick test — should start without errors
uv run gunicorn --bind 127.0.0.1:5001 --workers 1 app:app
# Ctrl+C to stop
```

### 2.5 Create a systemd service

Pick a **unique port** for this app (5001, 5002, 5003, etc.).

```bash
sudo nano /etc/systemd/system/your-app.service
```

Paste this (adjust the highlighted values):

```ini
[Unit]
Description=Your App Name
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/apps/your-app
EnvironmentFile=/home/ubuntu/apps/your-app/.env
ExecStart=/home/ubuntu/apps/your-app/.venv/bin/gunicorn \
    --bind 0.0.0.0:5001 \
    --workers 2 \
    --timeout 120 \
    app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> **For FastAPI**, change the last line of ExecStart to:
> ```
>     -k uvicorn.workers.UvicornWorker main:app
> ```
> (and make sure `uvicorn` is in your `pyproject.toml` dependencies)

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable your-app
sudo systemctl start your-app

# Check it's running
sudo systemctl status your-app
```

### 2.6 Point domain → app in Nginx Proxy Manager

1. **DNS**: Add an A record for `yourapp.example.com` → `YOUR_SERVER_IP`

2. **NPM UI** (`http://YOUR_SERVER_IP:81`):
   - **Proxy Hosts** → **Add Proxy Host**
   - Details tab:
     - Domain Names: `yourapp.example.com`
     - Scheme: `http`
     - Forward Hostname / IP: `172.17.0.1`
     - Forward Port: `5001`
   - SSL tab:
     - Request a new SSL Certificate ✓
     - Force SSL ✓
     - HTTP/2 Support ✓
     - Agree to Let's Encrypt ToS
   - Click **Save**

Done. `https://yourapp.example.com` is live.

**Note:** We use `172.17.0.1` (the Docker bridge gateway IP) instead of a hostname because Nginx inside the container uses its own DNS resolver, which doesn't read the container's `/etc/hosts` file. The direct IP bypasses DNS resolution entirely.

---

## Part 3: This App (ABAC_Sak)

A ready-made service file is included at `deploy/abac-app.service`.

```bash
# On the server
cd ~/apps
git clone <this-repo-url> ABAC_Sak
cd ABAC_Sak

uv sync

# Set up your .env
nano .env

# Install the systemd service
sudo cp deploy/abac-app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable abac-app
sudo systemctl start abac-app

# Verify
sudo systemctl status abac-app
```

Then add it in Nginx Proxy Manager: domain → `127.0.0.1:5001`.

---

## Part 4: Common Operations

### View logs
```bash
sudo journalctl -u your-app -f
```

### Restart after code changes
```bash
cd ~/apps/your-app
git pull
uv sync                          # in case dependencies changed
sudo systemctl restart your-app
```

### Stop an app
```bash
sudo systemctl stop your-app
```

### Disable an app (won't start on boot)
```bash
sudo systemctl disable your-app
```

### See all running app services
```bash
systemctl list-units --type=service --state=running | grep -v systemd
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Deploy new app | `git clone ... && uv sync && sudo systemctl enable --now app` |
| Update app | `git pull && uv sync && sudo systemctl restart app` |
| View logs | `sudo journalctl -u app -f` |
| Restart app | `sudo systemctl restart app` |
| Stop app | `sudo systemctl stop app` |
| Check status | `sudo systemctl status app` |
| Add domain + SSL | NPM UI → Add Proxy Host |

---

## Port Registry

Keep track of which app uses which port:

| Port | App | Domain |
|------|-----|--------|
| 5001 | ABAC_Sak | abac.example.com |
| 5002 | (next app) | ... |
| 5003 | ... | ... |

---

## Troubleshooting

### App won't start
```bash
sudo journalctl -u your-app --no-pager -n 50
```
Look for Python errors, missing env vars, or wrong paths.

### Domain not working
1. Check DNS: `dig yourapp.example.com` — should show your server IP
2. Check app is up: `curl http://127.0.0.1:5001` from the server
3. Check NPM proxy host config in the web UI

### SSL not issuing
- Port 80 must be open in AWS Security Group
- DNS must already be pointing to your server IP
- Wait a minute for DNS propagation

### Dependencies changed
```bash
cd ~/apps/your-app
uv sync
sudo systemctl restart your-app
```

### uv sync fails
```bash
# Check Python version requirement in pyproject.toml
uv python list
# Install a specific version if needed
uv python install 3.11
```
