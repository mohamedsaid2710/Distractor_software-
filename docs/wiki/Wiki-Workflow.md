# Wiki Workflow (No Repo/Wiki Confusion)

Use this workflow to keep source docs in the main repo while publishing to the GitHub Wiki safely.

## Rule of Separation

- Source of truth for docs: `docs/wiki/` in this repository.
- Published wiki: separate git repo (`*.wiki.git`).
- Do **not** copy the wiki clone inside this main repository.

## One-Command Sync

From the project root:

```bash
./scripts/wiki_sync.sh
```

By default this will:
1. Clone/update wiki repo in `/tmp/Distractor_software-wiki`
2. Sync `docs/wiki/*` -> wiki repo root
3. Show wiki repo git status

Then finalize publish:

```bash
cd /tmp/Distractor_software-wiki
git add .
git commit -m "Update wiki docs"
git push
```

## Custom Paths (Optional)

```bash
WIKI_DIR=/tmp/my-wiki-clone ./scripts/wiki_sync.sh
```

Custom wiki URL:

```bash
WIKI_URL=https://github.com/<user>/<repo>.wiki.git ./scripts/wiki_sync.sh
```

## Naming Conventions

- `Home.md` -> Wiki home page
- `_Sidebar.md` -> Left navigation in wiki
- Use page names that map cleanly in GitHub Wiki links:
  - `Usage.md` -> `[[Usage]]`
  - `Ibex-Integration.md` -> `[[Ibex Integration]]`

## Recommended Routine

1. Edit docs in `docs/wiki/*`.
2. Commit changes in main repo.
3. Run `./scripts/wiki_sync.sh`.
4. Commit/push the wiki repo.
