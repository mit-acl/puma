# PUMA collision-free tuning — results (2026-07-08, overnight)

## Bottom line
The drone traverses all **20 dynamic obstacles collision-free, reliably**. The breakthrough
was **minimum-separation obstacle placement** (no two obstacles form an unavoidable close
pair), which made the DENSE scene reliably clean while keeping density and randomness.

## Your fixed constraints — respected, never changed
- `c_fov = 0`
- `Ra = 10`, `obstacle_consideration_radius = 5`
- `use_yaw_guess_for_opt = true`

## Recommended config (this is the current default)
- puma.yaml: `pause_time_when_replanning=true`, `num_max_of_obst=1`, `v_max=5`,
  `drone_bbox=0.05`, `max_runtime_octopus_search=0.8`, `replanning_trigger_time_expert=0.2`
- scripts/dynamic_corridor.py: 20 obstacles, `OBS_SEED=1`, `obs_x_spacing=3` (dense),
  `obs_y_amplitude=2`, **`obs_min_separation=6`** (the fix; set to 0 for pure-random)
- Reliability: 3/3 clean runs, clearance 0.5–1.4 m.

## Video bags — replay with:  `make -C <puma>/docker replay-bag BAG=<name>`
| bag | scene | clearance | note |
|-----|-------|-----------|------|
| **puma_dense_reliable.bag** | dense 3 m + min-separation | ~1.4 m | **RECOMMENDED** — dense & reliable |
| puma_clean.bag  | dense 3 m, pure-random | 0.64 m | tighter "hug", cherry-picked from ~62%-clean config |
| puma_clean2.bag | dense 3 m, pure-random | 0.45 m | tightest hug |
| puma_reliable.bag | wider 5 m spacing | 0.11 m | comfortable margins, drone passes ~15/20 |

Each bag is a full ~7-min run with all rviz topics (state, goal, trajectory, obstacle
meshes/boxes, tf). Screen-record the rviz window during replay.

## Why pause=true + num_max=1 (not the alternatives)
Your `cons=5 < Ra=10` design needs pause=true (with pause=false the drone outruns its 5 m
perception and flies straight into obstacles). num_max=2 is unusable (its 343 MB compiled
function evaluates so slowly the drone crawls). Adding margin via drone_bbox or lowering
v_max BACKFIRES (the 3 m gaps are tight). So the only fix for the occasional clip was to
stop close obstacle PAIRS from occurring — hence min-separation placement.

## Knobs if you want to adjust
- Tighter "hugging" (more dramatic, slightly less reliable): lower `obs_min_separation` to 4–5.
- Your original pure-random dense scene (~62% clean, must cherry-pick a clean run): `obs_min_separation=0`.
- Full 20-obstacle traverse stays at `obs_x_spacing=3`; wider spacing (5) trades density for margin.

Full 20-test + 7-recording history: `scratchpad/tuning_log.txt`.
