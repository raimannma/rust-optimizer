---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git diff:*), Bash(git log:*), Bash(git config:*), Bash(git push:*), Bash(git branch:*), Bash(git rev-parse:*), Bash(gh pr:*)
description: Create a git commit, push, and open a PR
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Remote tracking: !`git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo "no upstream"`
- Recent commits (for style reference): !`git log --oneline -10`
- Author commits (for style reference): !`git log --author="$(git config user.email)" --oneline -10`
- All commits on this branch not on main: !`git log --oneline main..HEAD 2>/dev/null`
- User input (optional, can specify target branch): $ARGUMENTS

## Your task

Based on the above changes, create a git commit, push, and open a pull request.

1. If there are no staged changes, stage only the relevant changed files by name (never use `git add -A` or `git add .`).
2. Do NOT commit files that likely contain secrets (.env, credentials.json, etc).
3. Write a commit message following the guidelines below, then commit.
4. Push to the remote. Use `git push -u origin <branch>` if there is no upstream set.
5. Create a pull request using `gh pr create` targeting the base branch:
   - If the user specified a target branch in their input, use that.
   - Otherwise, default to `main`.
6. The PR title should match the commit title (or summarize all branch commits if multiple).
7. Write the PR body following the PR format below.
8. Return the PR URL when done.

## Commit message guidelines

- Use imperative mood (e.g., "add feature" not "added feature")
- First line: brief summary, max 72 characters
- Focus on the "why" and "what", not the "how"
- Be specific but concise
- Many commits do not need a body if the title is self-explanatory
  - Litmus test: "Would a developer understand this commit from the title + diff?" If yes, skip the body.
- Do NOT include "Generated with ...", "Co-Authored-By ...", or any AI attribution

### Conventional commit prefixes

Match the prefix to the nature of the change. These must align with `cliff.toml` commit parsers so they appear correctly in the changelog.

**Changelog: "Added"**
- `feat:` / `feat(scope):` — new feature

**Changelog: "Fixed"**
- `fix:` / `fix(scope):` — bug fix

**Changelog: "Changed"**
- `refactor:` / `refactor(scope):` — code restructuring without behavior change
- `perf:` / `perf(scope):` — performance improvement
- `docs:` / `docs(scope):` — documentation changes (README, guides, etc.)
- `style:` / `style(scope):` — formatting only (no logic change)
- `chore:` / `chore(scope):` — tooling, deps, config, scripts, AI config (.claude/, CLAUDE.md)

**Changelog: "Removed"**
- `revert:` / `revert(scope):` — revert a previous commit

**Excluded from changelog**
- `test:` / `test(scope):` — adding or updating tests
- `ci:` / `ci(scope):` — CI/CD pipeline changes

### Body rules

- When a body is needed (multiple important things in one commit), use bullet points.
- Body is separated from the title by a blank line.
- One bullet point per concept/change.
- Don't explain obvious things like "added unit tests for X".

### Commit format

Always pass the commit message via a HEREDOC:

```
git commit -m "$(cat <<'EOF'
<title line>

<optional body>
EOF
)"
```

## PR format

Use a HEREDOC for the body. Look at ALL commits on the branch (not just the latest) to write the summary.

```
gh pr create --title "<title>" --base <target-branch> --body "$(cat <<'EOF'
## Summary
<1-3 bullet points covering all branch commits>
EOF
)"
```

## Before you commit

1. Run `cargo +nightly fmt --all && cargo +nightly clippy --all-features --all-targets --fix --allow-dirty` to auto-format and fix lint issues.
2. Run all tests and doctests to ensure nothing is broken.
