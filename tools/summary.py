"""Print one table of every teacher and student evaluation in an output folder (no GPU needed).

Usage: python3 summary.py <output folder>      e.g. ~/spirob_runs/new_model
"""
import glob, json, os, sys

out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ".")
files = sorted(glob.glob(f"{out}/eval_*.json")) + sorted(glob.glob(f"{out}/student/eval_*.json"))
rows = []
for f in files:
    d = json.load(open(f))
    gr = d.get("groups", {})
    if not gr:
        continue
    b = {x["theta"]: x["success"] for x in d.get("by_angle", [])}
    rows.append((os.path.basename(f)[5:-5],
                 *(gr.get(k, {}).get("success", float("nan")) for k in ("nominal", "retention", "sector")),
                 *(b.get(t, float("nan")) for t in (50, 60, 70)), d.get("egg_err_mm")))
if not rows:
    sys.exit(f"no evaluations found in {out}")
print(f"{'evaluation':34s} {'nominal':>8s} {'retention':>10s} {'sector':>7s} {'50-60':>6s} {'60-70':>6s} {'70-80':>6s} {'egg err':>8s}")
for r in rows:
    cells = " ".join((f"{v:{w}.0%}" if v == v else f"{'-':>{w}}") for v, w in zip(r[1:7], (8, 10, 7, 6, 6, 6)))
    e = "-" if r[7] is None else f"{r[7]:.1f} mm"
    print(f"{r[0]:34s} {cells} {e:>8s}")
