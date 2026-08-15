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

Nothing on the Space is edited by hand. Every file it holds is written from
this repository, so the only way to change the demo is to push here.

1. A push to `main` touching `src/`, `app.py`, `pyproject.toml`,
   `.github/hf-space/`, or the sync script activates the workflow
2. `python -m scripts.hf_space --print-version` reads the version from
   `pyproject.toml`, and the job waits until PyPI has published it
3. The Space is cloned, and `python -m scripts.hf_space --space hf-space`
   writes three files into it:
   - `README.md` — the Space card, copied from `.github/hf-space/README.md`
   - `requirements.txt` — `turkic-translit==<version>`, and nothing else,
     because everything the demo needs is already a dependency of the package
   - `app.py` — the entry point named by the card's `app_file`
4. The commit is pushed, which makes Hugging Face rebuild
5. The job polls the Space's runtime API until the build settles, and **fails
   if it settles on an error**. A broken Space is a red workflow run, not a
   green one

### The pair that has to agree

A Space build installs the SDK version named on its card *alongside* the
packages in `requirements.txt`. So the card's `sdk_version` and the `gradio`
requirement in `pyproject.toml` are one setting written in two files, and when
they disagree pip refuses to install anything. That is what broke the demo:
the package moved to `gradio>=6.0,<7` while the card still said `5.29.0`.

Two checks now hold them together:

- `tests/test_hf_space.py` fails in `make check` when the card's SDK falls
  outside the requirement
- `scripts/hf_space.py` refuses to write the Space at all when it does

Raising the Gradio floor in `pyproject.toml` therefore means editing
`sdk_version` in `.github/hf-space/README.md` in the same commit. The test
names the pair that broke, so there is nothing to remember.

## Troubleshooting

**Space not updating?**
- Check the Actions tab: https://github.com/wagner-austin/turkic_transliteration/actions
- Verify your secrets are set correctly
- Make sure your HF token has write permissions

**Workflow failed at "Wait for the Space to build"?**
- The push landed; the build did not. The step prints Hugging Face's own
  error message, and the full log is under the Space's Logs → Build tab
- A dependency conflict there is nearly always the card-versus-pyproject pair
  described above, or a package the Space installs that the wheel cannot

**Want to update Space immediately after PyPI release?**
- Option 1: Push your version bump commit to main
- Option 2: Manually trigger the workflow (see above)
- Option 3: Add a step to your release script to push to main after PyPI upload
