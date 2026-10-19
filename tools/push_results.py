"""
push_results.py -- send small result files from a Kaggle/Colab run to the team's GitHub repo,
so results survive when the session ends and everyone can pull them.

In a notebook:
    from kaggle_secrets import UserSecretsClient
    os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")   # Add-ons -> Secrets
    !python push_results.py --repo OWNER/REPO --src /kaggle/working/results --dest results/p3_run1

Files over --max-mb are NOT pushed (put those in a Kaggle dataset instead).
The token is read from the environment and never printed.
"""
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path


def run(cmd, cwd=None, check=True):
    safe = cmd.replace(os.environ.get("GITHUB_TOKEN", "\0"), "***")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {safe}\n{(r.stdout + r.stderr).replace(os.environ.get('GITHUB_TOKEN', '\0'), '***')}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="OWNER/REPO, or a full git URL (used for testing)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True, help="folder inside the repo")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--max-mb", type=float, default=25.0)
    ap.add_argument("--message", default="results update")
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    url = a.repo if "://" in a.repo or a.repo.startswith("/") else (
        f"https://x-access-token:{token}@github.com/{a.repo}.git" if token else f"https://github.com/{a.repo}.git")

    src = Path(a.src)
    files = [p for p in src.rglob("*") if p.is_file()]
    keep = [p for p in files if p.stat().st_size <= a.max_mb * 1e6]
    for p in set(files) - set(keep):
        print(f"SKIP (>{a.max_mb} MB, use a Kaggle dataset): {p}")
    if not keep:
        print("nothing to push"); return 0

    work = Path(tempfile.mkdtemp())
    run(f"git clone --depth 1 --branch {a.branch} '{url}' repo", cwd=work)
    repo = work / "repo"
    run("git config user.email 'results@bot.local' && git config user.name 'results-bot'", cwd=repo)
    for p in keep:
        out = repo / a.dest / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)
    run(f"git add -A '{a.dest}'", cwd=repo)
    if run("git diff --cached --quiet", cwd=repo, check=False).returncode == 0:
        print("no changes"); return 0
    run(f"git commit -m '{a.message}'", cwd=repo)
    for attempt in range(3):                             # teammates may push at the same time
        run(f"git pull --rebase origin {a.branch}", cwd=repo, check=False)
        if run(f"git push origin HEAD:{a.branch}", cwd=repo, check=False).returncode == 0:
            print(f"pushed {len(keep)} file(s) to {a.dest}"); return 0
    raise RuntimeError("push failed 3 times")


if __name__ == "__main__":
    sys.exit(main())
