# Lab setup

How this checkout is set up for our rig: the Docker environment, the registered
cameras and grippers, and what is still uncalibrated. The upstream workflow is in
[README.md](README.md) and [docs/](docs/index.md); this file records only what is
specific to us.

## Hardware

| Role | Camera unit | Serial | Gripper unit | Gripper tag |
|---|---|---|---|---|
| Wrist | `insta360-X5-m0` | `IAHYA2509RQSP8` | `chopstick-m0` | ArUco 0 |
| Wrist | `insta360-X5-m1` | `IAHYA2509EVP5N` | `chopstick-m1` | ArUco 1 |
| Head (ego) | `insta360-ACEPRO2-001` | `IBGLH2601VVRXX` | - | hat tag ArUco 50 |

- Wrist cameras are named after the tag of the gripper they ride on.
- Grippers are printed from `../KIWI_hardware` (`chopsticks/`). Gripper tags are
  `DICT_4X4_50`, 14 mm; the hat tag is `DICT_4X4_100`, 50 mm (`--head-marker-id 50`).
- The hat module in KIWI_hardware is designed around the GO 3; the Ace Pro 2 needs
  its own mount. Head pose comes from the wrist cameras seeing the hat tag, so the
  tag only has to be rigid on the head.

## Docker environment

Everything runs through `docker compose` (`compose.yaml`, `docker/Dockerfile`); no
pixi on the host.

### One-time setup

```bash
cat > .env <<EOF
UID=$(id -u)
GID=$(id -g)
KIWI_ROOT=$PWD
DOCKER_GID=$(getent group docker | cut -d: -f3)
EOF

docker compose build                    # bakes the pixi envs into the image
git submodule update --init third_party/kalibr
docker build -t kalibr:latest -f third_party/kalibr/Dockerfile_ros1_20_04 third_party/kalibr
```

`.env` is gitignored. Optional: `PIXI_ENVIRONMENTS=default state viz tune mask-gpu`
to bake more environments (default: `default state viz tune`).

### Usage

```bash
docker compose run --rm pixi run test
docker compose run --rm pixi run -e state -- kiwi run --episode data/scene/demo
docker compose run --rm pixi run -e tune -- kiwi eef calibrate ...
docker compose run --rm pixi shell                       # interactive env shell
docker compose run --rm pixi-gpu run -e mask-gpu -- kiwi scene build --scene data/scene/scene.yaml
```

Any `pixi ...` command in the upstream docs becomes `docker compose run --rm pixi ...`.

### How it is wired

- **Envs are in the image.** Rebuild (`docker compose build`) after changing
  `pixi.toml` or `pixi.lock`; otherwise every run reinstalls into a throwaway
  container. `kiwi` is an editable install, so source edits need no rebuild.
- **The repo is mounted at its host path** (`KIWI_ROOT`), not `/workspace`.
  `kiwi calib` starts Kalibr with its own `docker run` on the host daemon, which
  resolves bind-mount paths on the host, so the paths must match.
- **Docker socket** (`/var/run/docker.sock`) is mounted for those Kalibr runs. This
  is root-equivalent access to the host; drop the line in `compose.yaml` if you do
  not calibrate from the container.
- **`/media` is mounted read-only**, so cameras plugged in over USB show up for
  `kiwi import`. Only the repo and `/media` are visible inside: put any file the
  container must read under the repo (e.g. `data/`, which is gitignored).
- **Host networking**, so viewers (viser, rerun, mkdocs serve) and the import
  picker (port 8765) are on `localhost`.
- **Runs as your UID**, so files written to `data/` and `build/` are yours.
- **X11 display** (`/tmp/.X11-unix` + your `$XAUTHORITY` cookie, `DISPLAY`
  from the host) for windowed tools such as `kiwi calib qr clock`. Needs an
  X11 session, not Wayland.
- The base image puts `/root/.pixi/bin` on `PATH`; the Dockerfile resets `PATH`
  because that directory is unreadable for a non-root user and turns every
  "command not found" into `EACCES`.

## Cameras

### Connecting a camera

Set the camera's USB mode to **U-Disk** (mass storage) and plug it in with USB-C. It
mounts at `/media/$USER/<label>/` with clips under `DCIM/Camera01/`. Pulling the
microSD into a card reader works the same way.

### Registering a new camera

KIWI binds an X5 recording to its unit by the serial embedded in the `.insv`. Read
it from any clip:

```bash
docker compose run --rm pixi run python -c "
from pathlib import Path
from kiwi.core.camera.registry import read_insv_serial, find_camera_by_serial
for c in sorted(Path('/media/$USER/Insta360 X5/DCIM/Camera01').glob('VID_*.insv')):
    s = read_insv_serial(c); print(c.name, s, find_camera_by_serial(Path('config/cameras'), s))
"
```

Then create `config/cameras/<unit>/camera.yaml` with `serial:`, `model:` and
`nickname:` (see [config/README.md](config/README.md)). The `.mp4` cameras (GO 3,
Ace Pro 2) are never bound by serial; name them at import (`--label ego_video`) and
at `kiwi init` (`--camera ego_video=<unit>`).

### Recording settings

| Camera | Mode | Must be off |
|---|---|---|
| X5 (wrists) | 4K 120 fps (1920x1920 per lens @ 119.88) | - |
| Ace Pro 2 (head) | Video mode (not FreeFrame), FOV **UltraWide**; **4K 119.88 fps** (3840x2160), pinned 2026-10-22 | stabilization (swipe up → Stabilization → Off); avoid the Dewarp / Linear / MegaView FOVs (lens-corrected) |

Keep a camera's mode fixed once it is calibrated: A/V delays and intrinsics are
stored per mode. Check a clip's mode with:

```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate -of csv=p=0 <clip>
```

The Ace Pro 2's capture settings are in its mp4 telemetry, readable with
`telemetry_parser.Parser(<clip>).telemetry()[0]["Default"]["Metadata"]`.

- `fov_type` reflects the FOV setting. The enum is in telemetry-parser's
  `src/insta360/extra_info.rs` (`ImageFovType`); the installed 0.3.0 predates
  some names and prints e.g. `Unknown15`, which is `FovTypeMega` (MegaView).
  The 2026-10-21 16:05 take was shot in MegaView, which the Insta360 manual
  says corrects distortion at the image edges.
- `is_dewarp` and `is_flowstate_online` are raw Insta360 protobuf fields
  (tags 43, 42) that neither Insta360 nor telemetry-parser documents. They
  stayed `True` (and `fov` stayed `105.0`) after switching to UltraWide with
  stabilization off, so do not use them as a check.
- The manual does not say whether UltraWide applies any lens correction.
  Checked from footage instead (2026-10-22, `VID_20261022_094334_004.mp4`,
  UltraWide, stabilization off): straight edges bow outward (barrel
  distortion, not rectilinear), and image motion follows the raw gyro -
  a 3-axis linear fit of median optical flow (central crop) on gyro rate,
  with a lag search, gives R^2 = 0.93 and image speed 0.9x the predicted.
  In-camera stabilization would cancel most of that motion.

## Calibration status

| Unit | Intrinsics / KB4 | Cam-IMU | IMU noise | A/V delay |
|---|---|---|---|---|
| `insta360-X5-m0` | borrowed (X5-002) | borrowed (X5-002) | borrowed (X5-002) | borrowed; cancels between two X5s in the same mode |
| `insta360-X5-m1` | borrowed (X5-002) | borrowed (X5-002) | borrowed (X5-002) | borrowed; cancels between two X5s in the same mode |
| `insta360-ACEPRO2-001` | pinhole-equi 3840x2160, 2026-10-22 (letter board; reproj 1.59 / 1.07 px) | 2026-10-22 (reproj median 1.48 px, -4.5 ms, IMU ~18 mm from optical centre) | none (standard priors) | measured 2026-10-22: -2.2 ms at 3840x2160@119.88 (3 runs) |

| Gripper | Camera-marker pose |
|---|---|
| `chopstick-m0` | calibrated 2026-10-21 (0.31 px, rigid to 0.4 mm / 1.0 deg p90) through borrowed intrinsics |
| `chopstick-m1` | calibrated 2026-10-21 (0.34 px, rigid to 0.4 mm / 1.0 deg p90) through borrowed intrinsics |

### Borrowed X5 calibration

We have no AprilGrid board yet, so both wrist X5s use verbatim copies of
`insta360-X5-002`'s calibration files (the files still say
`unit: insta360-X5-002`; each `camera.yaml` header says so). Across the shipped
X5s the principal point varies by ~5 px, the IMU position by ~2-4 mm and the
cam-IMU time offset by -4.8 to -7.6 ms, so expect somewhat worse tracking and
metric accuracy than a real calibration. `kiwi calib imu check` passes, but only
because it checks X5-002's own lenses against each other.

To borrow for another X5:

```bash
D=config/cameras/<new-unit>
for f in intrinsics.npz intrinsics-kb4.json intrinsics-kb4-back.json extrinsic-imu.yaml \
         extrinsic-imu-back.yaml imu-noise.yaml av-sync.yaml; do
    cp config/cameras/insta360-X5-002/$f $D/
done
# then add the matching pointer keys to $D/camera.yaml
```

### Replacing it with a real calibration

Needs the board: a **calib.io 10x7 AprilGrid, 50 mm tags, 15 mm spacing**
(`config/markers/calibio_kalibr_apriltag_board.yaml`), or a printed / on-screen grid
with its own marker config.

**Letter-size board (what we print in-house).** `data/board/aprilgrid_7x5_25mm_letter.pdf`:
7 x 5 tags of 25 mm, 7.5 mm gaps, ids 0-34, placed unscaled on a Letter
landscape page with a 100 mm scale bar (measured centre-to-centre of the end
ticks). Its config is `config/markers/letter_aprilgrid_7x5_25mm.yaml`; pass
`--marker-config config/markers/letter_aprilgrid_7x5_25mm.yaml` to every
`kiwi calib intrinsics run` / `imu cam`. Print at 100 % / Actual size, check
the bar, caliper a tag and write the measured edge into the config's
`tag_size`, then glue the sheet flat to something rigid (foam board, glass,
a clipboard). The board is small, so hold the lens close (roughly 15-30 cm) to
cover the fisheye periphery. To regenerate it, run
`kalibr_create_target_pdf --type apriltag --nx 7 --ny 5 --tsize 0.025 --tspace 0.3 --eps`
in the Kalibr image and place the EPS centred and unscaled on a 792 x 612 pt
page with Ghostscript (`-dFIXEDMEDIA`).

To print the full-size board, generate the PDF with the Kalibr image (`data/` is gitignored):

```bash
mkdir -p data/board
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD/data/board:/data -w /data \
    --entrypoint /bin/bash kalibr:latest -c 'source /catkin_ws/devel/setup.bash;
    rosrun kalibr kalibr_create_target_pdf --type apriltag --nx 10 --ny 7 \
        --tsize 0.05 --tspace 0.3 aprilgrid_10x7_50mm.pdf'
```

The page is 675 x 480 mm (grid 635 x 440 mm): print on A1 or a 24x36 in sheet
(A2 is too small) at 100 % / actual size, on matte paper, mounted flat on foam
board or dibond. Measure a tag with calipers afterwards (50.0 mm; gap 15.0 mm);
if it is off, put the measured size in a marker config rather than the nominal
one. All 70 ids (0-69) detect in the generated PDF with the config's
`DICT_APRILTAG_36h11`, `border_bits: 2`. Per lens, record 2-4 min at 4K 120 fps: 5 s still, 15 s
slow figure-8, 30 s rotations (+-30 deg per axis), 30 s translations (+-30 cm per
axis), 15 s fast shaking without blur, covering the fisheye periphery. Then follow
[docs/hardware.md](docs/hardware.md#cameras):

```bash
docker compose run --rm pixi run -- kiwi calib intrinsics run --insv <rec>.insv \
    --output-dir data/kalibr/<unit>_front --lenses front
docker compose run --rm pixi run -- kiwi calib intrinsics publish \
    --front-run data/kalibr/<unit>_front --camera <unit> --base <current intrinsics.npz>
docker compose run --rm pixi run -- kiwi calib intrinsics kb4 --camera <unit> --lens front
docker compose run --rm pixi run -- kiwi calib imu cam --insv <rec>.insv --camera <unit> \
    --lens front --output-dir data/kalibr/<unit>_imucam_front --frame-stride 20 --bag-from-to 5 95
```

Repeat for `--lens back`, run `kiwi calib imu check`, copy the camchain into the
unit directory and move the pointers. Afterwards, re-run the gripper calibration
with `--force`.

### Gripper calibration

Any clip with the gripper tag in the front lens works; the tag is rigid to the
camera even while the jaws move.

```bash
docker compose run --rm pixi run -e tune -- kiwi eef calibrate data/calib/<clip>.insv \
    --tool chopstick --gripper-unit chopstick-m0 --camera insta360-X5-m0 \
    --marker-id 0 --marker-size 0.014 [--dry-run] [--force]
```

Our two prints agree with each other (55.5 / 55.9 mm camera-to-tag) but sit ~5 mm
from the upstream authors' builds. Measure the printed tags with calipers: the
pose's scale follows `--marker-size`.

**Replaced upstream units.** KIWI forbids two builds of one design with the same
tag id (the tracker selects the tag by id alone), so our poses replaced the shipped
`chopstick-m0` / `chopstick-m1`. The README's public sample take was recorded with
the authors' builds; restore their files from git history to reproduce its
published numbers.

## Ace Pro 2 support

KIWI's docs assume a GO 3 for the ego view, but nothing in the pipeline is GO 3
specific: the ego camera is a generic single-lens `.mp4` stream, and head pose
does not use it. Changes made for the Ace Pro 2:

- `kiwi/core/camera/ingest.py`: the Ace Pro 2 names clips `VID_<date>_<time>_<seq>`
  (no `_00_` field). The clip-name patterns now accept that, so import reads its
  start time and LRV proxy, and an imported `..._ego_video.mp4` gets the
  `ego_video` label at `kiwi init`. Covered by
  `tests/episode/test_init.py::test_an_ace_pro_2_stem_without_the_middle_field_still_yields_its_tag`.
- `config/cameras/insta360-ACEPRO2-001/` registers the body (no calibrations).

### A/V delay (measured 2026-10-22)

Two different camera models need a measured A/V delay to sync by audio. Measured
against `insta360-X5-m1` front lens (1920x1920@119.88, the X5 reference zero) by
filming the rolling QR clock with both cameras, three runs, clapping at the start
of each. Inputs and reports are in `data/avsync/2026-10-22/`.

```bash
docker compose run --rm pixi run -- kiwi calib qr clock      # on this laptop; q / Esc closes
# per run: demux the X5 lens that saw the screen, then
ffmpeg -i X5.insv -map 0:v:0 -c:v copy -an X5_lens0.mp4
docker compose run --rm pixi run -- kiwi calib qr sync X5_lens0.mp4 ACE.mp4 \
    --left-undistort config/cameras/insta360-X5-m1/intrinsics.npz \
    --left-insv X5.insv --right-insv ACE.mp4 --audio --json-output runN/qr.json
docker compose run --rm pixi run -- kiwi calib qr av-delay X5.insv ACE.mp4 \
    --camera insta360-ACEPRO2-001 --qr-json runN/qr.json --run-label <date>-runN
```

| Run | QR video offset | Raw audio offset | A/V delay | Synced offset (`kiwi sync pair`) | Error vs QR |
|---|---|---|---|---|---|
| 1 | -311.7 ms | -309.3 ms | -2.4 ms | -311.5 ms | +0.2 ms |
| 2 | -253.7 ms | -246.5 ms | -7.1 ms | -248.7 ms | +5.0 ms |
| 3 | -157.1 ms | -160.0 ms | +2.9 ms | -162.2 ms | -5.1 ms |

Published as `av-sync-2026-10-22.yaml` (mean -2.2 ms, all audio gates passed). The
remaining run-to-run scatter is about +-5 ms, under one frame at 119.88 fps
(8.3 ms). The GO 3, for comparison, measured +62.5 ms. Repeating the runs adds
them to the average.
Intrinsics (`kiwi calib intrinsics run --model pinhole-equi`, mode-keyed like the
GO 3's) are only needed for the ego view in exports.

## To do

- [x] Letter AprilGrid printed (scale bar 100 mm) and its config added.
- [x] Ace Pro 2: stabilization off, FOV UltraWide, mode pinned to 4K 119.88 fps
      (2026-10-22); footage confirms no lens correction and no stabilization.
- [x] Ace Pro 2: A/V delay (-2.2 ms), intrinsics and cam-IMU (2026-10-22).
- [ ] **Deferred (2026-10-22): the X5s keep the borrowed X5-002 calibration.**
      To replace it: one letter-board recording per X5 lens (close, reach the
      edges, slow motion), solve as for the Ace Pro 2, then re-run both gripper
      calibrations with `--force`.
- [ ] Measure the printed gripper tags (should be 14.0 mm).
- [ ] Mount for the Ace Pro 2 + hat tag.
- [ ] IMU noise for each X5 (~2 h stationary recording, `kiwi calib imu noise`).
- [ ] First episode: record a fresh take with all three cameras in their pinned
      modes. The 2026-10-21 16:04 take predates the Ace Pro 2 settings (it was
      MegaView, edge-corrected), so its ego video does not match the
      calibration.
