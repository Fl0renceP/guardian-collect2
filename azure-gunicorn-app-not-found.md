# Bug: Azure Web App fails to start — `ModuleNotFoundError: No module named 'app'`

## Context

- **Project:** GuardianCollective (Discovery GradHack team project — Ctrl+Alt+Elite)
- **Stack:** Python/Flask backend (`backend/`), React/Vite frontend built to `frontend/dist/`
- **Deploy target:** Azure App Service on Linux, deployed via GitHub Actions (`azure/webapps-deploy@v3`)
- **Repo:** now public (GitHub Actions artifact storage quota issue was resolved by making the repo public — not the focus of this issue)
- **Recurring problem:** I've hit this exact same error on a *previous* Azure deploy too, so it's likely a systemic issue with how the startup command / app entrypoint is configured, not a one-off fluke.

## Current GitHub Actions workflow (`main_guardiancollective.yml`)

```yaml
name: Build and deploy Python app to Azure Web App - GuardianCollective

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python version
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Build frontend UI
        run: |
          cd frontend
          npm ci
          npm run build

      - name: Create and Start virtual environment and Install dependencies
        run: |
          cd backend
          python -m venv antenv
          source antenv/bin/activate
          pip install -r requirements.txt

      - name: Upload artifact for deployment jobs
        uses: actions/upload-artifact@v4
        with:
          name: python-app
          path: |
            .
            frontend/dist/**
            !backend/antenv/**

  deploy:
    runs-on: ubuntu-latest
    needs: build
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Download artifact from build job
        uses: actions/download-artifact@v4
        with:
          name: python-app

      - name: Login to Azure
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZUREAPPSERVICE_CLIENTID_33B979BC60B7447D8C76736FCB897A39 }}
          tenant-id: ${{ secrets.AZUREAPPSERVICE_TENANTID_CF3365AA09E143E4BFF874085D8CAF4D }}
          subscription-id: ${{ secrets.AZUREAPPSERVICE_SUBSCRIPTIONID_9D2734D5337F4802A464CF0B0E6D1601 }}

      - name: 'Deploy to Azure Web App'
        uses: azure/webapps-deploy@v3
        id: deploy-to-webapp
        with:
          app-name: 'GuardianCollective'
          slot-name: 'Production'
```

> Note: the artifact `path` currently includes the whole repo root (`.`), which means `backend/app.py` (or wherever the Flask entrypoint lives) gets deployed nested under a `backend/` subfolder rather than at the deployment root.

## Azure container startup log (relevant excerpt)

```
Python version : 3.13.13
App port : 8000

Launching oryx with: create-script -appPath /home/site/... -output /opt/startup/... -virtualEnvName antenv -defaultApp ...
Found build manifest file. Deserializing it...
Detected an app based on Django
Generating `gunicorn` command for 'backend.*'
Writing output script to '/opt/startup/...'
Using packages from virtual environment antenv located at /tmp/8dee.../antenv
Updated PYTHONPATH to '/opt/startup/...'

[INFO] Starting gunicorn 24.1.1
[INFO] Listening at: http://0.0.0.0:8000
[INFO] Using worker: sync
[INFO] Booting worker with pid: 1890
[ERROR] Exception in worker process
Traceback (most recent call last):
  File ".../arbiter.py", line 641, in spawn_worker
    worker.init_process()
  File ".../base.py", line 135, in init_process
    self.load_wsgi()
  File ".../base.py", line 147, in load_wsgi
    self.wsgi = self.app.wsgi()
  File ".../base.py", line 66, in wsgi
    self.callable = self.load()
  File ".../base.py", line 57, in load
    return self.load_wsgiapp()
  File ".../base.py", line 47, in load_wsgiapp
    return util.import_app(self.app_uri)
  File ".../util.py", line 366, in import_app
    mod = importlib.import_module(module)
  File ".../importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", ...
ModuleNotFoundError: No module named 'app'
[ERROR] Worker (pid:1890) exited with code 3.
[ERROR] Shutting down: Master
[ERROR] Reason: Worker failed to boot.
```

## Diagnosis

Two things are visibly in conflict:

1. **Oryx auto-detection** logged `Detected an app based on Django` and `Generating gunicorn command for 'backend.*'` — meaning Oryx guessed a Django-style entrypoint scoped to a `backend` package.
2. **The actual gunicorn error** is `from app import app` — which is the *default Flask-style* gunicorn invocation (`gunicorn app:app`), expecting a file called `app.py` at the working directory root with a Flask instance named `app`.

These two don't match, which is why the worker can't boot: gunicorn is trying to import a top-level `app` module, but the real Flask entrypoint most likely lives inside `backend/app.py` (or similarly nested), not at the deployment root — and the app is Flask, not Django, so Oryx's Django auto-detection may itself be wrong/misleading.

## What needs fixing

- [x] Confirmed Flask entrypoint: `backend/app.py` exports `app`.
- [x] Added a root-level shim at `app.py` that re-exports `backend.app:app` so default `gunicorn app:app` still resolves.
- [x] Added `backend/__init__.py` and hardened `backend/wsgi.py` to prefer `from backend.app import app`.
- [ ] Explicitly set the **Startup Command** in Azure App Service → Configuration → General settings, rather than relying on Oryx's auto-detected default. Recommended command:
  ```
  gunicorn --bind=0.0.0.0 --timeout 600 backend.app:app
  ```
  fallback command (if your site root differs):
  ```
  gunicorn --bind=0.0.0.0 --chdir backend --timeout 600 app:app
  ```
- [x] Verified `backend/requirements.txt` includes `gunicorn`.
- [x] Added an optional workflow step to set startup command via `az webapp config set` when `AZURE_RESOURCE_GROUP` secret is present.
- [ ] Double check whether Oryx's "Detected an app based on Django" auto-detection is being triggered by a stray `manage.py`, `settings.py`, or similar Django-looking file somewhere in `backend/` that shouldn't be there, or from residual artifacts of an earlier setup.

## Implemented in repo (this commit)

1. **`app.py` at repository root** now imports and re-exports `backend.app:app`.
2. **`backend/wsgi.py`** now prefers package-qualified import and falls back for local runs.
3. **`backend/__init__.py`** added so `backend` is always an importable package.
4. **Workflow automation (optional)** now sets Azure startup command during deploy if `AZURE_RESOURCE_GROUP` is configured as a GitHub secret.

These changes remove the fragile dependency on runtime working directory and make both `app:app` and `backend.app:app` resolvable in Azure App Service.

## Goal

Get gunicorn to correctly resolve and boot the Flask app instance so the container passes Azure's startup healthcheck, without relying on Oryx's default/auto-detected startup command, and set this up so it doesn't silently break again on the next deploy.
