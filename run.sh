#!/usr/bin/env bash
# SpiRob vision pipeline: one command per step. Settings are in config.env.
#   ./run.sh help        list the steps
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$HERE/config.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
TOOLS="$HERE/tools"
OUT="$RUNS/$MODEL_TAG"
LOGS="$SIR_REPO/logs/rsl_rl/spirob_minimal"
TEACHER0="$HERE/checkpoints/teacher_C1_a3.pt"
STUDENT0="$HERE/checkpoints/student_c1.pt"
C1="Mjlab-SpiRob-Polar-C1"
WIDE="Mjlab-SpiRob-Polar-Wide"
mkdir -p "$OUT"

py() { (cd "$SIR_REPO" && uv run python "$@"); }      # always run from sir's repo: his environment and model

# newest fine-tuned teacher of this model (highest checkpoint of the newest ft* run), else the starting teacher
current_teacher() {
    python3 - "$OUT" "$TEACHER0" <<'PY'
import glob, os, re, sys
out, fallback = sys.argv[1], sys.argv[2]
runs = sorted(glob.glob(f"{out}/ft*/"), key=os.path.getmtime)
for d in reversed(runs):
    c = sorted(glob.glob(f"{d}model_[0-9]*.pt"), key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))
    if c:
        print(c[-1]); break
else:
    print(fallback)
PY
}

eval_teacher() {   # <checkpoint> <task> <label>
    py "$TOOLS/polar_check.py" --mode eval --task "$2" --checkpoint "$1" --label "$3" \
       --out "$OUT" --num_envs "$NUM_ENVS_EVAL" --rounds 3
    echo "map: $OUT/eval_${3}_${2##*-}.png"
}

student() { py "$TOOLS/student_dagger.py" --out "$OUT/student" "$@"; }

step="${1:-help}"; shift || true
case "$step" in
  check)          # compatibility check of the robot model + checkpoints (~5 min)
    py "$TOOLS/check_model.py" --teacher "$TEACHER0" --student "$STUDENT0" ;;

  eval-teacher)   # [checkpoint] [task] [label]  -- default: current teacher on C1
    ck="${1:-$(current_teacher)}"; eval_teacher "$ck" "${2:-$C1}" "${3:-teacher_$(basename "$(dirname "$ck")")}" ;;

  finetune)       # <run name, e.g. ft1> [iterations=500] [task=C1]  -- continues from the current teacher
    run="${1:?give a run name, e.g. ./run.sh finetune ft1}"; iters="${2:-500}"; task="${3:-$C1}"
    init="$(current_teacher)"
    echo "fine-tuning $run on $task for $iters iterations, starting from $init"
    mkdir -p "$LOGS/cinit-$run" "$OUT/$run"
    cp "$init" "$LOGS/cinit-$run/model_init.pt"
    py "$TOOLS/train_polar.py" "$task" --env.scene.num-envs "$NUM_ENVS_TRAIN" --agent.max-iterations "$iters" \
       --agent.resume True --agent.load-run "cinit-$run" --agent.load-checkpoint model_init.pt \
       --agent.run-name "$run" --agent.logger tensorboard
    for d in "$LOGS"/*_"$run"; do cp "$d"/model_[0-9]*.pt "$OUT/$run/"; done
    ck="$(current_teacher)"; echo "newest checkpoint: $ck"
    eval_teacher "$ck" "$task" "$run" ;;

  coverage)       # current teacher on the whole wide sector (120-200 mm, 30-150 deg)
    eval_teacher "$(current_teacher)" "$WIDE" teacher_wide ;;

  student-check)  # camera images the student sees -> $OUT/student/camera_check.png
    student --mode check ;;

  eval-student)   # [student checkpoint] [label]  -- default: the starting student, no training
    student --mode eval --task "$C1" --student "${1:-$STUDENT0}" --label "${2:-student_start}" \
            --num_envs $((NUM_ENVS_EVAL / 2)) --eval_rounds 3 ;;

  dagger)         # [rounds=8]  -- DAgger from the starting student, labelled by the current teacher
    t="$(current_teacher)"; echo "teacher: $t"
    student --mode train --task "$C1" --teacher "$t" --init "$STUDENT0" \
            --num_envs $((NUM_ENVS_TRAIN / 2)) --rounds "${1:-8}" --betas 0.5,0.3,0.2,0.1,0 ;;

  ablations)      # [student checkpoint]  -- normal / black image / far-swap image
    s="${1:-$OUT/student/student_last.pt}"
    for abl in none black farswap; do
        student --mode eval --task "$C1" --student "$s" --label student_new --ablate "$abl" \
                --num_envs $((NUM_ENVS_EVAL / 2)) --eval_rounds 3
    done ;;

  summary)        # one table of all evaluations of this model (no GPU)
    python3 "$TOOLS/summary.py" "$OUT" ;;

  pipeline)       # everything in order, for a new model (several hours)
    "$0" check; "$0" eval-teacher "$TEACHER0" "$C1" teacher_start; "$0" finetune ft1
    "$0" coverage; "$0" eval-student; "$0" dagger; "$0" ablations; "$0" summary ;;

  help|*)
    cat <<TXT
SpiRob vision pipeline   (settings: config.env | model: $MODEL_TAG | outputs: $OUT)

  ./run.sh check                         compatibility check of the robot model + checkpoints (~5 min)
  ./run.sh eval-teacher [ckpt] [task]    evaluate a teacher (default: current teacher on C1)
  ./run.sh finetune ft1 [iters] [task]   fine-tune the current teacher (repeat with ft2, ft3 ... if needed)
  ./run.sh coverage                      current teacher on the whole wide sector
  ./run.sh student-check                 save the camera images the student sees
  ./run.sh eval-student [ckpt] [label]   evaluate a camera student (default: starting student)
  ./run.sh dagger [rounds]               retrain the student from the starting student with the current teacher
  ./run.sh ablations [ckpt]              student with normal / black / far-swap image
  ./run.sh summary                       table of all evaluations of this model
  ./run.sh pipeline                      all of the above in order

"current teacher" = newest fine-tuned checkpoint in $OUT/ft*/, otherwise checkpoints/teacher_C1_a3.pt
TXT
    ;;
esac
