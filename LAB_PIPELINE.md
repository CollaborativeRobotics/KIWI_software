# From recording to the Rerun view

Every step between "the take is recorded" and the Rerun view that shows the
videos, both wrist trajectories, the gripper opening, the hat marker, the base
tags and the YAM arms. Commands are the ones used for the 2026-10-23 tabletop
take (`data/tabletop/tabletop`); swap in your own scene and episode names.
Environment setup (Docker image, `.env`, Kalibr) is in
[LAB_SETUP.md](LAB_SETUP.md).

All `kiwi` commands run in the container, so each starts with
`docker compose run --rm`. Below, `$KIWI` stands for
`docker compose run --rm pixi run`:

```bash
KIWI="docker compose run --rm pixi run"
SCENE=data/tabletop          # one directory per place
EP=$SCENE/tabletop           # one directory per take
```

| Step | What | Command | Time (64 s take) |
|---|---|---|---|
| 1 | Import the three clips | `kiwi import` | ~30 s per camera |
| 2 | Write `episode.yaml` | `kiwi init` | seconds |
| 3 | Make the scene | `kiwi scene init` | seconds |
| 4 | Sync, demux, gripper | `kiwi run --profile prepare` | minutes |
| 5 | SLAM maps (GPU) | `kiwi scene build` | the longest step |
| 6 | Wrist states + hat | `kiwi run`, `--profile global-state-head` | minutes |
| 7 | YAM arms on the base tags | `kiwi replay yam --base-tags` | ~5 min first run, ~10 s after |
| 8 | Build and open the view | `kiwi viz play --save`, host `rerun` | first run makes proxies |
| 9 | Gaussian splat (optional) | `data/tools/` scripts, `kiwi real2simviz` | ~30 min GPU training |

## 1. Import

Plug in each camera in U-Disk mode, one at a time. `--no-pick` copies the
newest clip; leave it off to pick in the browser (port 8765) when the newest
clip is a false start.

```bash
# each wrist X5: named by its embedded serial (insta360-X5-m0 / -m1)
$KIWI -e state -- kiwi import --dest $EP/raw --no-pick
# the Ace Pro 2 carries no serial KIWI reads: label it
$KIWI -e state -- kiwi import --dest $EP/raw --no-pick --label ego_video
```

Check before moving on:

- **The pinned modes.** X5 1920x1920 @ 119.88; Ace Pro 2 3840x2160 @ 119.88:
  `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 <clip>`.
- **Ace Pro 2 FOV is UltraWide.** `fov_type` in
  `telemetry_parser.Parser(<clip>).telemetry()[0]["Default"]["Metadata"]`.
  Ignore its `is_dewarp` / `is_flowstate_online`; they read True regardless.
- **All three clips are the same take.** Their file names start at the same second.

## 2. Episode file

```bash
$KIWI -e state -- kiwi init --episode $EP \
    --left insta360-X5-m1 --right insta360-X5-m0 \
    --camera ego_video=insta360-ACEPRO2-001 --head-marker-id 50 --tool chopstick
```

Then edit `$EP/episode.yaml`:

- `description:` and `date:`
- `gripper_unit: chopstick-m1` under `left:`, `chopstick-m0` under `right:`
  (each camera is named after the gripper tag it rides on)
- `cameras.ego_video.head_marker.size: 0.05`: `kiwi init` writes its 0.049
  default; the hat tag is 50 mm

## 3. Scene

A new place gets its own scene, built from one mapping episode. The mapping
episode needs at least 60 s of usable video.

```bash
$KIWI -e state -- kiwi scene init --root $SCENE --mapping-episode tabletop
```

For another take in a place that already has a scene, put the episode under
that scene's root and re-run `kiwi scene init` there, then skip step 5.

## 4. Sync, demux, gripper

```bash
$KIWI -e state -- kiwi run --episode $EP --profile prepare
```

Check:

- **Sync** printed an offset per stream and `loop <= 0.1 ms`.
- **eef** valid-frame fraction per arm. The first take had ~90 %; this take's
  left arm had 67 %, so a third of its gripper opening is interpolated. Low
  values usually mean the gripper tag was out of view or blurred.

## 5. SLAM maps (GPU)

```bash
docker compose run --rm pixi-gpu run -e mask-gpu -- kiwi scene build --scene $SCENE/scene.yaml
```

Builds three ORB-SLAM3 maps (front-left, front-right, shared rear) and aligns
them into one metric, gravity-aligned scene frame.

- **Keep the laptop awake** while anything uses the GPU: a suspend with a live
  CUDA context hangs the job and needs a reboot.
- **Check each map's alignment.** A map can build and still be wrong. On this
  take the front-right map came out internally inconsistent: transformed into
  the scene, its keyframes spanned ~2 cm while the right wrist moved ~39 cm.
  The fused wrist states were still fine, because they do not depend on that
  map alone. To catch it, compare each map's keyframe centres in the scene
  frame with the matching wrist's `state.tum` extent. The keyframes come from
  `data/tools/atlas_dump`; the transform is `s·R·x + t` from
  `scene_map/scene_frame.json`. A longer, slower pass over the table with each
  hand is the usual fix.

## 6. Wrist states and the hat

```bash
$KIWI -e state -- kiwi run --episode $EP                               # sync, convert, state, eef
$KIWI -e state -- kiwi run --episode $EP --profile global-state-head   # + head marker
```

- `state` writes `<arm>/scene/state.tum`: each wrist's front-camera pose in the
  scene frame, on the episode clock.
- `head` writes `head/head_marker.tum`: the hat tag's pose wherever a wrist
  rear camera saw it. On this take that was 1,708 poses covering 45.6 of 64 s.
  It is the tag, not the Ace Pro 2's lens. Placing the ego camera needs the
  hat-tag-to-lens calibration, which is not in the repo yet.

## 7. YAM arms on the base tags

ArUco tags 52 and 53 (`DICT_4X4_100`, 50 mm) on the table mark the two YAM
bases.

```bash
$KIWI -e replay -- kiwi replay yam $EP --base-tags 52 53 --no-render
```

1. **Locates the tags** in the four wrist lens videos and places them in the
   world through the wrist trajectories. The result is cached in
   `$EP/base_tags.json`. Pass `--redetect` after re-running the state stage;
   the cache is only keyed on the ids and size.
2. **Places one base on each tag.** The arms go on the side of the line
   between the tags where the hands worked, each base yawed toward its own
   hand's workspace, with up from gravity.
3. **Solves the IK** and writes `replay_yam.npz` (joints, tracking error,
   issues) and `replay_yam_world.npz` (the arm links in the world, for the
   viewers).

Check the printed summary:

- **Each tag's detections and spread.** On this take: 53 at 9 mm, 52 at 19 mm.
- **Which tag became which base,** and whether a `note: a hand worked nearer
  the other arm's base` appears.
- **Tracking error and issue time.** On this take: 5.2 mm mean, 4.1 s of IK
  miss. Without the tags (`--ws-yaw 60`, centred) it was 55 mm and 44 s.

Drop `--no-render` to also write `replay_yam.mp4`, rendered on the CPU.
`kiwi replay yam-view $EP` plays the rollout in the browser, but that needs
WebGL.

## 8. The Rerun view

`kiwi viz play` cannot open a window from inside the container (the image has
no X11 libraries, and it still exits 0). Save a recording in the container
and open it with a Rerun viewer on the host.

### One-time viewer setup (host)

1. **Install `uv`**, if `uv --version` fails. It fetches the Rerun viewer on
   first use and caches it.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh     # installs to ~/.local/bin
   ```

2. **Create the ffmpeg shim.** On Linux the Rerun viewer decodes H.264 with
   the host's `ffmpeg`, and needs 5.1 or newer. Ubuntu 22.04 ships 4.4, which
   leaves every video panel showing an error. The shim runs the KIWI image's
   ffmpeg 9 instead, so nothing new is installed on the host:

   ```bash
   mkdir -p ~/.local/opt/kiwi-ffmpeg
   cat > ~/.local/opt/kiwi-ffmpeg/ffmpeg <<'EOF'
   #!/usr/bin/env bash
   # Run the kiwi-pixi image's ffmpeg for the host Rerun viewer (needs >= 5.1).
   # Rerun pipes the bitstream through stdin/stdout, so no mounts are needed.
   exec docker run --rm -i --network none \
       --entrypoint /home/kiwi/envs/kiwi-6875959163789674014/envs/state/bin/ffmpeg \
       kiwi-pixi "$@"
   EOF
   chmod +x ~/.local/opt/kiwi-ffmpeg/ffmpeg
   ~/.local/opt/kiwi-ffmpeg/ffmpeg -version | head -1     # expect "ffmpeg version 9..."
   ```

   The env path inside the image changes when the image is rebuilt. If the
   check fails, find the new path and edit the shim:
   `docker compose run --rm --entrypoint bash pixi -c 'ls /home/kiwi/envs/*/envs/state/bin/ffmpeg'`.
   Each video starts its own decoder container, so the first frames take a
   second or two to appear. On a host with ffmpeg 5.1+ already installed,
   skip the shim and the `PATH=` prefix below.

3. **Use the container's Rerun version.** The host viewer must match the SDK
   that wrote the file, 0.36.3 today. Check after a `pixi.lock` update with
   `$KIWI -e viz -- python -c "import rerun; print(rerun.__version__)"`.

The window needs an X11 session (`echo $XDG_SESSION_TYPE` prints `x11`), as
for the other windowed tools in [LAB_SETUP.md](LAB_SETUP.md).

### Every time

```bash
$KIWI -e viz -- kiwi viz play $EP --save $EP/$(basename $EP).rrd
PATH=$HOME/.local/opt/kiwi-ffmpeg:$PATH uvx --from rerun-sdk==0.36.3 rerun $EP/$(basename $EP).rrd
```

- The first `viz play` makes small H.264 copies of each lens (`proxy.mp4`);
  later runs reuse them. Re-run `viz play --save` after any stage that changes
  what is drawn (state, head, `replay yam`); the `.rrd` is a snapshot.
- Each `rerun` launch streams into a viewer already running on port 9876. If
  the recordings panel then lists two, pick the newer one, or close the
  viewer first.

What the view shows (summary panel, top right):

| Layer | From | Entity |
|---|---|---|
| video | `<stream>/proxy.mp4` | `streams/<label>` |
| trajectory | `<arm>/scene/state.tum` | `world/path/<arm>`, `world/camera/<arm>` |
| tool state | `<arm>/eef.npz` | `tool_state/<arm>` |
| head | `head/head_marker.tum` | `world/path/head_marker`, `world/head/head_marker` |
| base tags | `base_tags.json` | `world/base_tags/<id>` |
| robot | `replay_yam_world.npz` | `world/robot/yam` |

Robot links are tinted where the rollout has a problem: red for an IK miss,
magenta for a joint jump, yellow for a tip jump.

## 9. Gaussian splat (optional)

A 3DGUT Gaussian splat of the scene, viewed with `kiwi real2simviz`, which
can also draw the wrist paths and the YAM arms. It needs steps 1–6. Step 7 is
needed only for the arms.

KIWI's release does not include the splat builder. The scripts used here are
in `data/tools/`, which is **gitignored**: copy them somewhere tracked before
relying on them.

| Script | Does |
|---|---|
| `data/tools/atlas_dump/build/atlas_dump` | dumps an ORB-SLAM3 atlas's map points and keyframes as text |
| `data/tools/splat/build_dataset.py` | builds a COLMAP-format training set from the episode, with masks and seed points |
| `data/tools/splat/train_and_package.sh` | trains 3DGUT and writes the run records `real2simviz` checks |

```bash
ASSET=data/real2sim/tabletop-asset
```

### One-time setup

```bash
docker compose build real2sim                                     # CUDA 12.8 image
docker compose run --rm real2sim bash third_party/setup_real2sim.sh   # 3DGRUT into build/3dgrut/.venv
```

`atlas_dump` is built against the staged ORB-SLAM3 tree
(`third_party/build_orbslam3.sh`, then CMake in `data/tools/atlas_dump` with
`-DORBSLAM_SRC=$PWD/build/orbslam3/src`), and runs in the `pixi` image.

### 9a. Dump the SLAM map points

Seed points for the splat come from the three maps built in step 5.

```bash
mkdir -p $SCENE/atlas_dump
docker compose run --rm --entrypoint bash pixi -c "
S=$SCENE/scene_map; O=$SCENE/atlas_dump
for role in front_left front_right shared_rear; do
  m=\$(/home/kiwi/envs/*/envs/default/bin/python -c \"import json; print(json.load(open('\$S/current.json'))['roles']['\$role'])\")
  data/tools/atlas_dump/build/atlas_dump build/orbslam3/src/Vocabulary/ORBvoc.txt \$S/\$m/atlas.osa \$O/\$role
done"
```

Each run prints `maps N points P keyframes K` and writes
`<role>_points.txt` (`map x y z n_obs`) and `<role>_keyframes.txt`
(`map kf_id t x y z`), in the map's own units. Check that `maps` is 1; a
multi-map atlas would need its other maps handled.

### 9b. Move the points into the scene frame

Each map's similarity transform is in `scene_map/scene_frame.json`:
scene point = `s · R · x + t`.

```bash
$KIWI python -c "
import json, numpy as np
d = '$SCENE/atlas_dump'
fr = json.load(open('$SCENE/scene_map/scene_frame.json'))['maps']
for role, m in fr.items():
    P = np.loadtxt(f'{d}/{role}_points.txt')
    X = m['scale'] * P[:, 1:4] @ np.array(m['rotation']).T + np.array(m['translation'])
    np.save(f'{d}/{role}_scene.npy', np.c_[X, P[:, 4]])
    print(role, len(X), 'p5-p95 extent', np.percentile(X, [5, 95], axis=0).round(2).tolist())
"
```

**Check the extents**, and the keyframe check from step 5. A map whose
points collapse to a few centimetres is misaligned, and its seeds would be
wrong. On the tabletop take, front-right was: its seeds were swapped for an
empty set, keeping the original file aside:

```bash
mv $SCENE/atlas_dump/front_right_scene.npy $SCENE/atlas_dump/front_right_scene.published_transform.npy
$KIWI python -c "import numpy as np; np.save('$SCENE/atlas_dump/front_right_scene.npy', np.zeros((0, 4)))"
```

This only removes seed points. All four lens streams are still used as
training images, because their poses come from the fused wrist states, not
from any single map.

### 9c. Build the training set (GPU)

```bash
docker compose run --rm real2sim build/3dgrut/.venv/bin/python data/tools/splat/build_dataset.py \
    --episode $EP --asset $ASSET --atlas-points $SCENE/atlas_dump
```

Writes `$ASSET/dataset/` and `$ASSET/trajectories/`, using these defaults:

- the sharpest frame per 1 s from each of the four lenses, at 960 px;
- fisheye (KB4) intrinsics;
- frames turning faster than 90 °/s dropped;
- pixels more than 85° off-axis ignored;
- masks over the gripper, the camera-fixed rig and people (DeepLabV3 on the GPU).

The tabletop take gave 252 images and 10,034 seed points. The 143 s first
take gave 566 images.

### 9d. Train (GPU, ~30 min)

**Keep the laptop awake and on power.** A suspend hangs training, the
container ends up "Dead", and only a reboot frees the GPU. If it happens,
`docker stop -t 20 kiwi-splat-train`, reboot, and rerun this step; 9a–9c
need no redo.

```bash
docker compose run --rm --name kiwi-splat-train real2sim \
    bash data/tools/splat/train_and_package.sh $ASSET           # optional 2nd arg: iterations (30000)
```

It trains for 30,000 iterations with the sample asset's settings, then writes
`$ASSET/training/run-001/run.json` and `$ASSET/run.json`. A successful run
ends with `packaged ... checkpoint training/run-001/...`.

Do not trust the exit status alone: it can be 0 after a hung run is stopped.
Check that `$ASSET/run.json` exists and says `"status": "completed"`.

### 9e. View

```bash
docker compose run --rm real2sim build/3dgrut/.venv/bin/kiwi real2simviz $ASSET --port 8895 \
    --robot $EP            # optional: the YAM arms from step 7
```

Open `http://127.0.0.1:8895`: the left hand is blue, the right green. This
viewer is viser in the browser, so it **needs WebGL**. Where the browser has
none, render saved views to PNG instead:

```bash
docker compose run --rm real2sim build/3dgrut/.venv/bin/python data/real2sim/make_views.py $ASSET <out_dir>   # writes stills.json, orbit.json
docker compose run --rm real2sim build/3dgrut/.venv/bin/kiwi real2simviz $ASSET \
    --views <out_dir>/stills.json --output <out_dir>/stills
```

`make_views.py` (also in gitignored `data/`) places five stills along the
take and a 72-step orbit around the hands.

Status as of 2026-10-25: the first scene's asset (`data/real2sim/first-asset`)
is complete. The tabletop training was cut off by a suspend and still has to
be rerun (step 9d).
