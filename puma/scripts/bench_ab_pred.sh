#!/bin/bash
# RUN THIS ON THE MACHINE WITH A DISPLAY (needs the depth camera to render).
# Runs PANTHER* then PUMA, both WITH prediction, and saves a bag for each.
set -e
PUMA=/home/kkondo/code/puma_ws/src/puma/puma
DEMO=$PUMA/other/demos/uncertainty_aware_planner_demo.py
BAGDIR=$PUMA/other/demos/data/bags
cd "$PUMA"
sed -i -E 's/USE_PERFECT_PREDICTION = "[a-z]+"/USE_PERFECT_PREDICTION = "false"/' "$DEMO"   # WITH prediction
run_one() {  # agent cfov outbag
  sed -i -E 's/AGENTS_TYPES = \["[^"]*"\]/AGENTS_TYPES = ["'"$1"'"]/' "$DEMO"
  sed -i -E 's/^c_fov: [0-9.]+/c_fov: '"$2"'/' "$PUMA"/param/puma.yaml
  echo ">>> Running $1 (c_fov=$2) WITH prediction. Watch rviz; it ends at the goal (or times out)."
  rm -f "$BAGDIR"/run.bag "$BAGDIR"/run.bag.active
  make -C "$PUMA"/docker run-planner-record || true
  sleep 2
  [ -f "$BAGDIR"/run.bag ]        && mv -f "$BAGDIR"/run.bag        "$BAGDIR"/"$3"
  [ -f "$BAGDIR"/run.bag.active ] && mv -f "$BAGDIR"/run.bag.active "$BAGDIR"/"$3".active
  echo ">>> saved $3"
}
run_one parm_star 1.0 panther_star_pred.bag
run_one puma      0.0 puma_pred.bag
echo "DONE. Tell Claude to extract metrics from panther_star_pred.bag and puma_pred.bag."
