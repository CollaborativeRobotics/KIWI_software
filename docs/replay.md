# Replay and Gaussian scenes

## Replay

`kiwi replay` replays an episode's tool-tip trajectories kinematically on two FR3 arms (URDF/MuJoCo, no ROS) and reports whether a given robot placement can follow them.

```bash
pixi run -e replay kiwi replay solve data/ep1 data/ep2     # batch IK, metrics, <episode>/replay_solve.npz
pixi run -e replay kiwi replay run data/ep1 data/ep2       # interactive viewer
pixi run -e replay kiwi replay view fr3_chopstick_sa       # inspect a robot description
```

Each episode needs a `replay_state.npz` (`kiwi export state`); `solve` builds a missing or outdated one.
`solve` prints per-episode tracking error, IK failures, collisions and continuity, and writes the joint trajectories and settings to `replay_solve.npz`.
The default rig places the two arm bases 0.805 m apart; `--workspace-x/y/z`, `--ws-yaw/pitch/roll`, `--base-*`, `--ik-*`, `--offset` and `--traj-yaw` change it.
All episodes of one run come from the same scene and map set.

The viewer plays the solved trajectories and lets the bases, workspace, gripper mount and joints be moved, with IK re-solved on demand.

### Robot descriptions

| Model | Use |
| --- | --- |
| `chopstick_dc` | handheld capture gripper; its URDF gives export the camera-to-tip transform |
| `chopstick_sa` | servo-actuated gripper for the robot |
| `fr3_chopstick_sa`, `fr3_dual_chopstick_sa` | FR3 with the SA gripper, single and dual arm |

Sources and generated files live in `descriptions/` (see [PROVENANCE](https://github.com/TToTMooN/KIWI_software/blob/main/descriptions/PROVENANCE.md)).
After changing a source, regenerate the URDF and MuJoCo models with `pixi run -e replay python -m kiwi.replay.generate` (`--check` verifies the generated files).

The DC `camera_lens0` frame is not a calibration; export needs a calibrated `camera_lens0` in the source URDF or a `build.mount_cam` override in the gripper unit.
Replay maps the normalized opening onto the SA jaw's CAD travel `[-0.872665, 0]` rad.

### YAM rollout

`kiwi replay yam` rolls episodes out on two 6-dof YAM arms whose bases are 1 m apart along Y, and renders each one to `<episode>/replay_yam.mp4` on the CPU. The solved joints go to `<episode>/replay_yam.npz`.

```bash
pixi run -e replay kiwi replay yam data/ep1                  # one episode
pixi run -e replay kiwi replay yam                           # synthetic demo -> build/replay_yam_demo.mp4
pixi run -e replay kiwi replay yam data/ep1 --separation 0.8 --workspace 0.3 0 0.1
```

`kiwi replay yam-view data/ep1` plays a solved `replay_yam.npz` in the browser (viser, port 8080), with a time slider, play/speed/loop and frame-stepping controls.

The rig (`descriptions/yam/urdf/yam_dual_chopstick_sa.urdf`) takes its joint origins and limits from the YAM xacro in `bmm_robot_description`. Each link is a cylinder, and a simplified Chopstick SA is mounted on each flange using the SA's CAD flange-to-TCP offset.
The IK tracks the tool tips in the episode's shared frame. By default the mean tool-tip position is placed at `--workspace`; pass `--no-center` to place the episode's world origin there instead.
The YAM's reach is short, so check the tracking error the command prints. You can tune `--workspace`, `--ws-yaw` and `--mount-yaw` to bring an episode into reach.
Every rollout is checked for three kinds of problem, and the command prints a summary of them:

| Issue | Flagged when | Colour |
| --- | --- | --- |
| Joint jump / flip | one joint steps faster than `--joint-speed` (10 rad/s, the YAM limit) | magenta, on the link that joint drives |
| IK miss | tool-tip error above `--ik-error-mm` (20 mm) | red, whole arm, with a line to the missed target |
| Tip jump | the achieved tool tip moves faster than `--tip-speed` (1 m/s); labelled by whether the recording itself jumped | yellow, whole arm |

The video tints the affected arms and links in these colours and adds a timeline strip that shows every flagged section. `yam-view` adds an **Issues** folder with the same strip, a list of sections you can jump to, previous/next buttons, and sliders for the thresholds. `real2simviz --robot` tints the arms in the splat and lists the sections under **Robot issues**.

After changing the generator, regenerate the URDF with `python -m kiwi.replay.yam --regenerate` (`--check` reports whether the checked-in file is out of date).

## Gaussian scenes

`kiwi real2simviz` renders a static, metric Gaussian scene asset with NVIDIA's [3DGRUT](https://github.com/nv-tlabs/3dgrut) renderer and overlays the recorded wrist trajectories.

### Install

Linux x86-64 with an NVIDIA GPU, the CUDA 12.8 toolkit, a C++ compiler, Git, curl and uv:

```bash
bash third_party/setup_real2sim.sh        # --cuda-home <path> if CUDA 12.8 is not on PATH
```

The renderer is installed in `build/3dgrut/.venv` (`KIWI_3DGRUT` or `--backend-repo` selects another location).

### View

```bash
pixi run -e state -- kiwi real2simviz <asset> --port 8895
pixi run -e state -- kiwi real2simviz <asset> --episodes data/room/ep3 --port 8895   # another processed episode of the same scene
pixi run -e state -- kiwi real2simviz <asset> --views views.json --output renders    # render saved views to PNG
```

`--robot data/room/ep3` draws that episode's YAM rollout into the scene. It reads `replay_yam_world.npz`, which `kiwi replay yam` writes alongside its other outputs. The arms are placed in the episode's world at the workspace transform the rollout was solved with, follow the timeline, and are hidden wherever the reconstructed scene is in front of them; **Show robot arms** toggles them.

The command runs from any KIWI environment and starts the renderer inside `build/3dgrut/.venv`.

Open `http://127.0.0.1:8895`; the left hand is blue and the right green.
**Save PNG** and **Save view** write to the asset's `viewer/` directory.

### Asset

```text
<asset>/
  run.json        entry point: names the dataset, the checkpoint and the trajectories
  dataset/        images, cameras, sparse points and manifest.json, whose digest the checkpoint's training run.json must match
  training/       the 3DGRUT checkpoint and its training run.json
  trajectories/   left/right trajectories and fixed transforms
  viewer/         captures and saved views, created on first save
```

Assets are produced by KIWI's scene reconstruction pipeline, which is not part of this release. One asset is published: the scene of the public sample take from the README, with that take's wrist trajectories ([Drive](https://drive.google.com/file/d/1IlMvveFabSqR9wCnaKE8YnfTOmEG0ARk/view), 509 MB, SHA-256 `7a41144ee862a82afba5ae8da68f942301ba23d3b73942fed0d65fefdab17ee9`):

```bash
gdown 1IlMvveFabSqR9wCnaKE8YnfTOmEG0ARk -O kiwi-night-scene-asset.tar.gz
sha256sum kiwi-night-scene-asset.tar.gz
tar xzf kiwi-night-scene-asset.tar.gz
pixi run -e state -- kiwi real2simviz kiwi-night-scene-asset --port 8895
```
