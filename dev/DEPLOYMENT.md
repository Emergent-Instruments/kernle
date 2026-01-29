# Kernle Deployment Guide

This document covers deployment procedures for all Kernle components.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Kernle Platform                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PyPI Package (kernle)          pip install kernle         │
│  ├─ CLI: kernle                                            │
│  └─ MCP Server: kernle-mcp                                 │
│                                                             │
│  Backend API (Railway)          api.kernle.ai              │
│  └─ FastAPI + Supabase                                     │
│                                                             │
│  Documentation (Mintlify)       docs.kernle.ai             │
│  └─ docs-site/                                             │
│                                                             │
│  Web Dashboard (Vercel)         kernle.ai                  │
│  └─ web/ (Next.js)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Reference

| Component | Platform | Domain | Deploy Command |
|-----------|----------|--------|----------------|
| Python Package | PyPI | pypi.org/project/kernle | `make publish` |
| Backend API | Railway | api.kernle.ai | Auto-deploy on push |
| Docs Site | Mintlify | docs.kernle.ai | `mintlify deploy` |
| Web Dashboard | Vercel | kernle.ai | Auto-deploy on push |

---

## 1. GitHub Workflow

### Branch Strategy

```
main          Production-ready code (auto-deploys to all platforms)
├─ feature/*  New features (PR to main)
├─ fix/*      Bug fixes (PR to main)
└─ docs/*     Documentation updates (PR to main)
```

### Creating a Release

1. **Update version numbers:**
   ```bash
   # Update pyproject.toml
   # version = "0.2.0"

   # Update backend/pyproject.toml if applicable
   ```

2. **Create changelog entry:**
   ```bash
   # Add to CHANGELOG.md (if exists) or commit message
   ```

3. **Commit and tag:**
   ```bash
   git add .
   git commit -m "chore: bump version to 0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main --tags
   ```

4. **Create GitHub Release:**
   - Go to GitHub → Releases → Draft new release
   - Select the tag
   - Generate release notes
   - Publish release

### Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code formatted (`black .` and `ruff check .`)
- [ ] Documentation updated if needed
- [ ] No secrets or credentials in code
- [ ] Changelog/commit message describes changes

---

## 2. PyPI Deployment

### Prerequisites

```bash
# Install build tools
pip install build twine

# Configure PyPI credentials
# Option 1: Use keyring
pip install keyring
keyring set https://upload.pypi.org/legacy/ __token__

# Option 2: Use ~/.pypirc
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE
EOF
chmod 600 ~/.pypirc
```

### Build and Publish

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI first (recommended)
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ kernle

# Upload to PyPI (production)
twine upload dist/*
```

### Verify Installation

```bash
# Install fresh
pip install --upgrade kernle

# Verify version
kernle --version

# Test basic functionality
kernle -a test-deploy status
```

### Makefile Commands

```makefile
# Add to Makefile
.PHONY: build publish publish-test

build:
	rm -rf dist/ build/ *.egg-info
	python -m build

publish-test: build
	twine check dist/*
	twine upload --repository testpypi dist/*

publish: build
	twine check dist/*
	twine upload dist/*
```

---

## 3. Railway Deployment (Backend API)

### Initial Setup

1. **Create Railway Project:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli

   # Login
   railway login

   # Link project
   cd backend/
   railway link
   ```

2. **Configure Environment Variables:**

   In Railway dashboard → Variables:
   ```
   SUPABASE_URL=https://xxx.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=eyJ...
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

3. **Deploy:**
   ```bash
   # Manual deploy
   railway up

   # Or push to main (auto-deploys if connected)
   git push origin main
   ```

### Railway Configuration

File: `backend/railway.json`
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

### Health Check

```bash
# Check deployment status
curl https://api.kernle.ai/health

# Expected response:
# {"status": "healthy", "version": "0.1.0"}
```

### Logs and Debugging

```bash
# View logs
railway logs

# View logs in real-time
railway logs --follow

# Open Railway dashboard
railway open
```

### Rollback

```bash
# List deployments
railway deployments

# Rollback to previous deployment
railway rollback
```

---

## 4. Vercel Deployment (Web Dashboard)

### Initial Setup

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Link Project:**
   ```bash
   cd web/
   vercel link
   ```

3. **Configure Environment Variables:**

   In Vercel dashboard → Settings → Environment Variables:
   ```
   NEXT_PUBLIC_API_URL=https://api.kernle.ai
   NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
   ```

### Deploy

```bash
# Preview deployment
vercel

# Production deployment
vercel --prod

# Or connect GitHub for auto-deploy
# Vercel dashboard → Import Project → Select repo
```

### Vercel Configuration

Create `web/vercel.json`:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "regions": ["sfo1"],
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "no-store" }
      ]
    }
  ]
}
```

### Custom Domain

1. Go to Vercel dashboard → Domains
2. Add domain: `kernle.ai`
3. Configure DNS:
   ```
   A     @     76.76.21.21
   CNAME www   cname.vercel-dns.com
   ```

---

## 5. Mintlify Deployment (Documentation)

### Initial Setup

1. **Install Mintlify CLI:**
   ```bash
   npm install -g mintlify
   ```

2. **Test Locally:**
   ```bash
   cd docs-site/
   mintlify dev
   # Opens at http://localhost:3000
   ```

### Deploy

```bash
# Deploy to Mintlify
cd docs-site/
mintlify deploy

# Or connect GitHub for auto-deploy
# Mintlify dashboard → Connect Repository
```

### Custom Domain

1. In Mintlify dashboard → Settings → Custom Domain
2. Add domain: `docs.kernle.ai`
3. Configure DNS:
   ```
   CNAME docs  proxy.mintlify.com
   ```

### Configuration

File: `docs-site/mint.json`
```json
{
  "name": "Kernle",
  "logo": { "dark": "/logo/dark.svg", "light": "/logo/light.svg" },
  "favicon": "/favicon.png",
  "colors": { "primary": "#10B981" },
  "navigation": [...]
}
```

---

## 6. Environment Variables

### Local Development

Create `.env` in project root:
```bash
# Agent Configuration
KERNLE_AGENT_ID=dev-agent

# Supabase (optional, for cloud features)
KERNLE_SUPABASE_URL=https://xxx.supabase.co
KERNLE_SUPABASE_KEY=eyJ...

# Logging
KERNLE_LOG_LEVEL=DEBUG
```

### Backend (.env)

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Server
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO

# Optional: Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Web Frontend (.env.local)

```bash
NEXT_PUBLIC_API_URL=https://api.kernle.ai
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```

---

## 7. CI/CD Setup (GitHub Actions)

### Test Workflow

Create `.github/workflows/test.yml`:
```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest tests/ -v --tb=short

      - name: Lint
        run: |
          ruff check .
          black --check .
```

### PyPI Publish Workflow

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### Docs Deploy Workflow

Create `.github/workflows/docs.yml`:
```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
    paths:
      - 'docs-site/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Mintlify
        run: npm install -g mintlify

      - name: Deploy
        working-directory: docs-site
        env:
          MINTLIFY_TOKEN: ${{ secrets.MINTLIFY_TOKEN }}
        run: mintlify deploy
```

---

## 8. Deployment Checklist

### Before Release

- [ ] All tests pass (`pytest tests/`)
- [ ] Code linted (`ruff check . && black --check .`)
- [ ] Version bumped in `pyproject.toml`
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
- [ ] No hardcoded secrets or credentials
- [ ] Environment variables documented

### PyPI Release

- [ ] Build succeeds (`python -m build`)
- [ ] Package checks pass (`twine check dist/*`)
- [ ] TestPyPI upload successful (optional)
- [ ] PyPI upload successful
- [ ] Fresh install works (`pip install --upgrade kernle`)
- [ ] CLI commands work (`kernle --version`)

### Backend Release

- [ ] Docker builds locally
- [ ] Health check endpoint responds
- [ ] Railway deployment successful
- [ ] Production health check passes
- [ ] No error spikes in logs

### Frontend Release

- [ ] Build succeeds locally (`npm run build`)
- [ ] Vercel preview deployment works
- [ ] Production deployment successful
- [ ] All pages load correctly
- [ ] API connections work

### Docs Release

- [ ] Local preview looks correct (`mintlify dev`)
- [ ] All links work
- [ ] Search index updated
- [ ] Custom domain resolves

---

## 9. Troubleshooting

### PyPI Issues

**"Invalid distribution" error:**
```bash
# Clean and rebuild
rm -rf dist/ build/ *.egg-info
python -m build
```

**"File already exists" error:**
```bash
# Version already published - bump version in pyproject.toml
```

### Railway Issues

**Deployment fails:**
```bash
# Check logs
railway logs

# Verify Dockerfile
docker build -t kernle-backend backend/
docker run -p 8000:8000 kernle-backend
```

**Health check fails:**
```bash
# Ensure /health endpoint exists and responds quickly
curl http://localhost:8000/health
```

### Vercel Issues

**Build fails:**
```bash
# Check build locally
cd web/
npm run build
```

**Environment variables not loading:**
- Verify variables in Vercel dashboard
- Redeploy after adding variables

### Mintlify Issues

**Deploy fails:**
```bash
# Validate mint.json
cd docs-site/
mintlify dev  # Should show any config errors
```

---

## 10. Contacts and Resources

| Resource | Link |
|----------|------|
| GitHub Repo | https://github.com/Emergent-Instruments/kernle |
| Railway Dashboard | https://railway.app/project/kernle |
| Vercel Dashboard | https://vercel.com/emergent-instruments/kernle |
| Mintlify Dashboard | https://dashboard.mintlify.com |
| PyPI Package | https://pypi.org/project/kernle |
| Production API | https://api.kernle.ai |
| Documentation | https://docs.kernle.ai |
| Web Dashboard | https://kernle.ai |
