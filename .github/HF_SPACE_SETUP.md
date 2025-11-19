# Hugging Face Space Auto-Sync Setup

This repository automatically syncs to the Hugging Face Space after pushing changes to the `main` branch.

## Setup Instructions

### 1. Get your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token" or use an existing token with **write** access
3. Copy the token (starts with `hf_...`)

### 2. Add GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your GitHub repo: https://github.com/wagner-austin/turkic_transliteration/settings/secrets/actions
2. Click "New repository secret"
3. Add these two secrets:

   **Secret 1: HF_TOKEN**
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token (e.g., `hf_xxxxxxxxxxxxx`)

   **Secret 2: HF_USERNAME**
   - Name: `HF_USERNAME`
   - Value: `AustinWagner` (your HF username)

### 3. That's it!

The workflow will now automatically:
- Trigger when you push code changes or version updates to `main`
- Extract the version from `pyproject.toml`
- Clone your HF Space repository
- Push an update to trigger a rebuild
- Your Space will automatically pull the latest version from PyPI

## Manual Trigger

You can also manually trigger the sync from GitHub:
1. Go to: https://github.com/wagner-austin/turkic_transliteration/actions/workflows/sync-to-hf-space.yml
2. Click "Run workflow"
3. Select the `main` branch
4. Click "Run workflow"

## How It Works

1. When you push to `main` (e.g., after running your release script), the GitHub Action activates
2. It reads your current version from `pyproject.toml`
3. It clones your HF Space repo: `AustinWagner/turkic-transliteration-demo`
4. It makes a commit to trigger a rebuild
5. It pushes to HF, which triggers Hugging Face to rebuild the Space
6. The Space's `requirements.txt` has `turkic-translit[ui]>=0.3.2`, so it pulls the latest version from PyPI

## Troubleshooting

**Space not updating?**
- Check the Actions tab: https://github.com/wagner-austin/turkic_transliteration/actions
- Verify your secrets are set correctly
- Make sure your HF token has write permissions

**Want to update Space immediately after PyPI release?**
- Option 1: Push your version bump commit to main
- Option 2: Manually trigger the workflow (see above)
- Option 3: Add a step to your release script to push to main after PyPI upload
