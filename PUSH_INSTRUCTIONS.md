# Publishing this repository to GitHub (to be run by you)

Before pushing, check:

- [ ] Read `docs/VALIDATION_LIMITS.md` and the available validation reports.
- [ ] No data, credentials or personal paths in the folder. Run the checks in step 2.
- [ ] `LICENSE` is included (GNU GPLv3).

## 1. Create an empty public repository on GitHub

On github.com: **New repository**, name `prc2026-strong-turtle`, **Public**, *no* README, *no* licence and *no* .gitignore (we provide them).

## 2. Safety checks (PowerShell, in the repo folder)

```powershell
Set-Location 'PATH_TO_REPOSITORY'
python check_source.py
```

The source check must pass. It omits ignored local data and submission folders.

## 3. Check the licence

```powershell
Get-Content -LiteralPath LICENSE -TotalCount 3
```

## 4. Commit and push

Replace `YOUR_GITHUB_USERNAME`:

```powershell
git init
git add .
git status
git commit -m "PRC Data Challenge 2026 - team strong-turtle final solution"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/prc2026-strong-turtle.git
git push -u origin main
```

Local Git has already been initialized on `main` in the prepared copy; `git init` above is harmless if repeated. If Git requests a commit identity, set it for this repository and repeat the commit:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_COMMIT_EMAIL"
```

`git status` must list source, documentation and licence files only. Data, work files and submissions are excluded by `.gitignore`. Keep aggregate CSV/JSON validation reports in `docs/`; they contain no per-flight predictions.

## 5. Share the link

Send the repository URL to the organisers as requested.

Publishing code does not upload competition predictions. The final local prediction file is `submission/strong-turtle_final.parquet`; its integrity checks and SHA-256 are recorded in `docs/candidate_manifest.json`. The reviewed recipe retains self-training and excludes base distillation after the nested February/June results. The separate June replica repair probe is not part of this candidate.
