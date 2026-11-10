# From a solved rollout to the YAM arms: sim, then the real robot

After [LAB_PIPELINE.md](LAB_PIPELINE.md) step 7 (`kiwi replay yam`), an episode has
`replay_yam.npz`: per-frame joints for both YAM arms and how well each frame
was solved. This page turns that into motion: a joint CSV, a rehearsal in
Isaac through the same ROS topics as the real followers, and then the real
arms. Every step is per episode, so it is the same for every dataset.

| Step | Where | Command |
|---|---|---|
| 1 | KIWI container | `kiwi replay yam-export $EP` |
| 2 | anywhere with numpy + tyro | `replay_trajectory.py --csv joints_50hz.csv` (plan only) |
| 3 | GPU host with isaac_yam (optional) | `scripts/run_sim6.sh replay_joint_csv.py` |
| 4 | GPU host with isaac_yam + sr-evals | `scripts/rehearse_ros_replay.sh up` / `replay` / `down` |
| 5 | robot station | `yams.launch.py`, then `replay_trajectory.py --execute --ui` |

The code is in three repos:

- **KIWI_software:** `kiwi replay yam-export` (`kiwi/replay/yam_export.py`).
- **openarm_ros2:** `openarm_openpi/scripts/replay_trajectory.py`, with the web UI in
  `openarm_openpi/scripts/common/replay_ui.py` and `replay_ui.html`.
- **isaac_yam:** `scripts/replay_joint_csv.py` and `scripts/rehearse_ros_replay.sh`. They
  need `scripts/ros_sim_station.py`, which is on the `jontoto/sim-ros2-station` branch.

The ROS bridge is sr-evals `sim_station/ros_bridge_node.py`.

## 1. Export the joint CSV

```bash
$KIWI -e replay -- kiwi replay yam-export $EP
```

This writes `$EP/joints_50hz.csv` (50 Hz, the YAM control rate) and
`$EP/joints_50hz.json`, which is a report.

**The CSV:**
- Columns 1-14 are in the YAM `DESIRED_JOINT_ORDER`: left joint1-6, left
  gripper opening (m), right joint1-6, right gripper.
- Columns 15-16 are `left_ok` and `right_ok`. A step is 1 when that arm's
  solve is usable:
  - tool-tip error ≤ 20 mm;
  - no tip or joint jump;
  - inside the joint limits to `--limit-tol` (0.02 rad).
- Joints are clipped to the limits both the real arm and Isaac's model accept.

**The report** lists each arm's unsolved stretches and how much was clipped.
Check them before going further: a long unsolved stretch on one arm splits
the trajectory into sections.

## 2. Check the plan

```bash
python3 openarm_openpi/scripts/replay_trajectory.py --csv joints_50hz.csv
```

This moves nothing. It prints the **sections**: runs of at least 5 s
(`--min-section-s`) where every commanded arm's solve is ok, with 0.3 s
trimmed off each end. For each section it also prints:
- its peak joint speed;
- how far apart consecutive sections are;
- how many steps get clamped to the limits.

Playback defaults:
- half speed (`--speed 0.5`);
- joints rate-limited to 1.5 rad/s (`--max-joint-vel`);
- moves to each section start at 0.3 rad/s peak (`--transition-vel`).

To replay one arm only, pass `--arms LEFT` or `--arms RIGHT`.

## 3. Isaac video (optional)

This drives the sim directly, without ROS, and writes
`replay.mp4` (left wrist | overhead | right wrist) and the per-joint
tracking error:

```bash
ISAAC_YAM_ARM_LAYOUT=aloha scripts/run_sim6.sh replay_joint_csv.py --csv joints_50hz.csv --out-dir out
```

On a shared GPU, RTX frames that come out as flat grey mean VRAM ran out.
Free some VRAM, or set `IY_SIM_DEVICE=cpu`, which runs physics on the CPU
(the cameras still need the GPU).

## 4. Rehearse the real-robot replay in Isaac

`rehearse_ros_replay.sh`, in isaac_yam, runs three things:
- the Isaac station, as a stand-in robot;
- the sr-evals ROS bridge, which publishes the followers' topics and cameras;
- the followers' gravity feedforward.

All three run on their own DDS domain (91) and ports, so they can sit next
to other stations. The replay script then talks to that stack exactly as it
would to the robot:

```bash
scripts/rehearse_ros_replay.sh up                      # ~1 min to load the scene
OPENARM_SCRIPTS=<openarm_ros2>/openarm_openpi/scripts \
  scripts/rehearse_ros_replay.sh replay $EP/joints_50hz.csv [--speed 0.3 ...]
scripts/rehearse_ros_replay.sh down
```

Open the web UI at `http://localhost:8642/`. From another computer, first run
`ssh -L 8642:localhost:8642 <gpu host>`. To reach the replay's terminal, run
`tmux attach -t kiwi-replay`.

Logs and the commanded/measured joint log (`replay_<time>.npz`) go to
`~/kiwi_replay/run`.

`OPENARM_SCRIPTS` mounts an openarm_ros2 checkout's scripts into the image.
You need it until the image ships `replay_trajectory.py`.

The Isaac arms use the ALOHA layout, not the episode's base tags. The joint
motion is the real one, but where the hands end up relative to the table is
not.

## 5. The real robot

1. **Check the bases.** The bases must stand where step 7 of the pipeline
   placed them: on the base tags, at the yaws it printed. Joint-space replay
   puts the hands where the demo had them only if that holds.
2. **Bring up the followers.** This includes their gravity feedforward:

   ```bash
   ros2 launch openarm_openpi yams.launch.py mode:=inference station:=<station>
   ```
3. **Start the replay:**

   ```bash
   ros2 run openarm_openpi replay_trajectory.py --csv joints_50hz.csv --execute --ui [--ui-cameras cam_high]
   ```

The UI listens on the station's localhost only, port 8642, so reach it from
your laptop through `ssh -L`. The header shows which computer the page
controls.

Each section runs in four steps:
1. *Move to start*: a smooth minimum-jerk move to the section's first pose.
2. The arms hold there.
3. *Play*: the section plays until its solve ends.
4. The next section's move waits for you.

At any time:
- **STOP**: the button or Space; Ctrl-C in the terminal does the same. The
  arms decelerate along their path over 0.3 s and hold. You can then:
  - *Resume*;
  - *Restart this section*;
  - *Next section*;
  - *Start over*;
  - *Quit*.
- **Go here:** every section has a *Go here* button, so you can jump to it.
  Restarts and jumps still stop at that section's *Move to start* prompt.
- **After the last section**, the script waits so you can start over.

**If the page loses contact:** a motion stops by itself when the page hasn't
updated for 2 s (`--ui-deadman-s`), for example because the tab was closed or
the network dropped. Keep the tab in front while the arms move.

**Grippers and limits:**
- Grippers are never commanded.
- Every command is clamped to the joint limits.

The page is not an E-stop. Keep the hardware stop at hand.
