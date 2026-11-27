# Replay, in order, every change the old session made to its files: Write, Edit, full Read (ground truth) and the
# Python snippets run through Bash that rewrote a file (snippets/*.py, reviewed: each one only reads a file, replaces
# strings and writes it back). Everything happens inside <root>, which mirrors the old paths:
#   /Users/davide/Desktop/NN_ResearchProject                                   -> <root>/repo
#   /Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/memory -> <root>/old_memory
# Usage: python3 -I replay.py <recovery dir> <snippets dir> <root>
import base64
import json
import os
import re
import subprocess
import sys

MAP = [("/Users/davide/Desktop/NN_ResearchProject", "repo"),
       ("/Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/memory", "old_memory"),
       ("/Users/davide/.claude/plans", "old_plans")]
NUMBERED = re.compile(r"^ *(\d+)\t(.*)$")


def local(root, path):
    for old, new in MAP:
        if path == old or path.startswith(old + "/"):
            return os.path.join(root, new + path[len(old):])
    return None


def read_lines(text):
    out = {}
    for raw in text.split("\n"):
        m = NUMBERED.match(raw)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def main(rec, snip_dir, root):
    uses = [json.loads(l) for l in open(os.path.join(rec, "Jrk_tool_uses.jsonl"))]
    results = {}
    for l in open(os.path.join(rec, "Jrk_tool_results.jsonl")):
        r = json.loads(l)
        results[r["tool_use_id"]] = r
    snippets = {s["k"]: s for s in json.load(open(os.path.join(snip_dir, "index.json")))}
    log = []
    checks = []
    for k, u in enumerate(uses):
        r = results.get(u["tool_use_id"])
        if r is None or r.get("is_error"):
            continue
        inp = u.get("wire_input") or u["input"]
        when = u["created_at"][:19]
        if u["name"] in ("Write", "Edit", "MultiEdit", "Read"):
            path = local(root, inp.get("file_path", ""))
            if path is None:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            exists = os.path.exists(path)
            if u["name"] == "Write":
                open(path, "w").write(inp["content"])
                log.append((when, path, "write"))
            elif u["name"] == "Read":
                c = r["content"]
                if path.endswith(".png"):
                    try:
                        open(path, "wb").write(base64.b64decode(json.loads(c)["source"]["data"]))
                        log.append((when, path, "image"))
                    except Exception as e:  # noqa: BLE001
                        log.append((when, path, f"image error {e}"))
                    continue
                lines = read_lines(c)
                if not lines:
                    continue
                n = max(lines)
                full = inp.get("offset") in (None, 0, 1) and inp.get("limit") is None and min(lines) == 1
                long_line = any(len(v) >= 2000 for v in lines.values())
                if exists:  # compare with the replayed state
                    cur = open(path).read().split("\n")
                    bad = [i for i, v in lines.items() if i - 1 >= len(cur) or cur[i - 1] != v]
                    checks.append((when, path, f"lines {min(lines)}-{n}", len(bad)))
                if full and not long_line and "PARTIAL view" not in c:
                    open(path, "w").write("\n".join(lines.get(i, "") for i in range(1, n + 1)) + "\n")
                    log.append((when, path, f"read full {n}"))
            else:
                if not exists:
                    log.append((when, path, "EDIT ON MISSING FILE"))
                    continue
                t = open(path).read()
                for e in (inp["edits"] if u["name"] == "MultiEdit" else [inp]):
                    a, b = e["old_string"], e["new_string"]
                    cnt = t.count(a)
                    if cnt == 0:
                        log.append((when, path, "EDIT OLD_STRING NOT FOUND"))
                        continue
                    t = t.replace(a, b) if e.get("replace_all") else t.replace(a, b, 1)
                    log.append((when, path, "edit" if cnt == 1 or e.get("replace_all") else f"edit ambiguous x{cnt}"))
                open(path, "w").write(t)
        elif u["name"] == "Bash" and k in snippets:
            s = snippets[k]
            cwd = local(root, s["cwd"])
            os.makedirs(cwd, exist_ok=True)
            code = open(os.path.join(snip_dir, s["file"])).read()
            p = subprocess.run([sys.executable, "-I", "-"], input=code, text=True, cwd=cwd, capture_output=True)
            status = "ok" if p.returncode == 0 else "FAILED: " + (p.stderr.strip().splitlines() or [""])[-1][:200]
            log.append((when, s["file"], f"snippet {status}"))
    json.dump({"log": log, "checks": checks}, open(os.path.join(root, "replay_log.json"), "w"), indent=1)
    for row in log:
        if "NOT FOUND" in row[2] or "MISSING" in row[2] or "FAILED" in row[2] or "ambiguous" in row[2]:
            print("PROBLEM", row)
    for row in checks:
        if row[3]:
            print("READ MISMATCH", row)
    print("events", len(log), "read checks", len(checks))


if __name__ == "__main__":
    main(*sys.argv[1:4])
