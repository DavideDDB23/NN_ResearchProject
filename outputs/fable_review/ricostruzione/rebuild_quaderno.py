# Rebuild outputs/fable_review/quaderno.md of the lost Mac from the old session: the base is the awk print of
# 14 Nov 11:10 (X.01 to the end), the Reads of 14 Nov 20:31 (lines 196-275 and 329-488) fill and check the middle,
# then every Edit and the Python snippet of 15 Nov 19:51 are applied in order. The Reads of 15 Nov 14:31
# (1144-end) and 16:24 (1498-1513) are used as checks. Lines 1-195 (P1.01-P1.13) were never printed: a note says so.
# Usage: python3 -I rebuild_quaderno.py <recovery dir> <snippets dir> <out file>
import json
import os
import re
import subprocess
import sys
import tempfile

NUMBERED = re.compile(r"^ *(\d+)\t(.*)$")
HOLE = "@@RIGHE_1_195_NON_RECUPERATE@@"


def numbered(text):
    out = {}
    for raw in text.split("\n"):
        m = NUMBERED.match(raw)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def main(rec, snip_dir, out_file):
    uses = [json.loads(l) for l in open(os.path.join(rec, "Jrk_tool_uses.jsonl"))]
    res = {}
    for l in open(os.path.join(rec, "Jrk_tool_results.jsonl")):
        r = json.loads(l)
        res[r["tool_use_id"]] = r
    snippets = {s["k"]: s for s in json.load(open(os.path.join(snip_dir, "index.json")))}
    doc = None
    headings_early = []
    for k, u in enumerate(uses):
        inp, r = u["input"], res.get(u["tool_use_id"])
        if r is None or r.get("is_error"):
            continue
        when = u["created_at"][:19]
        cmd = inp.get("command", "") if u["name"] == "Bash" else ""
        if doc is None and "awk '/^## X" in cmd:
            out = r["content"]
            heads, body = out.split("\n----\n", 1)
            headings_early = [h for h in heads.split("\n") if re.match(r"^\d+:#", h)]
            doc = HOLE + "\n" + body.rstrip("\n") + "\n"
            print(when, "base from awk:", len(body.splitlines()), "lines")
            continue
        if doc is None:
            continue
        if u["name"] == "Read" and inp.get("file_path", "").endswith("quaderno.md"):
            lines = numbered(r["content"])
            a, b = min(lines), max(lines)
            block = "\n".join(lines[i] for i in range(a, b + 1))
            if a == 196:  # fills the hole up to the X.01 section (line 274)
                head = "\n".join(lines[i] for i in range(196, 274))
                tail = "\n".join(lines[i] for i in range(274, b + 1))
                assert doc.split(HOLE + "\n", 1)[1].startswith(tail), "lines 274-275 differ"
                doc = doc.replace(HOLE + "\n", HOLE + "\n" + head + "\n", 1)
                print(when, "filled lines 196-273")
            else:
                # a check: the block must appear as it is; the last line can be cut by the limit
                body = "\n".join(lines[i] for i in range(a, b))
                print(when, f"check lines {a}-{b}:", "OK" if body in doc else "MISMATCH")
        elif u["name"] == "Edit" and inp.get("file_path", "").endswith("quaderno.md"):
            old, new = inp["old_string"], inp["new_string"]
            n = doc.count(old)
            if n != 1 and not inp.get("replace_all"):
                print(when, f"EDIT PROBLEM count={n}", repr(old[:60]))
                if n == 0:
                    continue
            doc = doc.replace(old, new) if inp.get("replace_all") else doc.replace(old, new, 1)
        elif u["name"] == "Bash" and k in snippets and "quaderno" in open(os.path.join(snip_dir, snippets[k]["file"])).read():
            with tempfile.TemporaryDirectory() as d:
                open(os.path.join(d, "quaderno.md"), "w").write(doc)
                code = open(os.path.join(snip_dir, snippets[k]["file"])).read()
                p = subprocess.run([sys.executable, "-I", "-"], input=code, text=True, cwd=d, capture_output=True)
                print(when, "snippet", snippets[k]["file"], "exit", p.returncode, p.stdout.strip()[:80], p.stderr.strip()[-200:])
                doc = open(os.path.join(d, "quaderno.md")).read()
    note = ("> **Nota della ricostruzione (26 novembre 2026).** Il quaderno originale è andato perso con il Mac. Questa\n"
            "> versione è ricostruita dalle chiamate della sessione Claude del 14–16 novembre. Le righe 1–195 (intestazione\n"
            "> e voci da P1.01 a P1.13 della revisione dell'11–13 novembre) non sono mai state stampate in quella sessione e\n"
            "> non si possono recuperare: i loro esiti sono in `docs/findings.md` (sezione \"Review of blocks 1 to 3\").\n"
            "> I titoli noti di quelle voci, con il numero di riga originale:\n>\n"
            + "".join(f">     {h}\n" for h in headings_early if int(h.split(':')[0]) < 196))
    doc = doc.replace(HOLE + "\n", "# Quaderno di laboratorio\n\n" + note + "\n", 1)
    open(out_file, "w").write(doc)
    print("written", out_file, len(doc.splitlines()), "lines")


if __name__ == "__main__":
    main(*sys.argv[1:4])
