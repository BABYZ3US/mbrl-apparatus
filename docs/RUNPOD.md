# Running training campaigns on RunPod

The full loop: provision a GPU pod → get the repo on it → bootstrap the env →
launch a campaign → monitor → pull the verdict → stop. All the scripts live in
`scripts/`; this doc is the operator runbook + the hard-won gotchas.

> **The golden rule:** W&B has every metric the moment it logs, independent of
> the pod. If the pod dies/reassigns mid-run, you lose nothing that already
> ran — pull the verdict from W&B (§6b). Checkpoints on the `/workspace` volume
> make any interrupted arm resumable from its last save.

---

## 0. One-time setup (do this once, saves endless pain)

1. **Add your SSH public key to RunPod account-level** (Settings → SSH Public
   Keys), not just per-pod. RunPod injects account keys into every pod at boot,
   so restarts/new pods don't lock you out.
   ```bash
   cat ~/.ssh/id_ed25519.pub        # generate first if needed: ssh-keygen -t ed25519
   ```
2. **W&B key**: grab it from wandb.ai/settings (academic account on the .edu email).

---

## 1. Provision the pod

- **GPU**: RTX 3090 or 4090 (24 GB). These nets are tiny (<3 GB VRAM) and
  **compute-bound** — a single card saturates at ~4 concurrent arms; the win
  from a 2-GPU pod is concurrency, not per-arm speed. Don't pay for A100/H100.
  Prioritize **high vCPU count** (MuJoCo env stepping is CPU-bound — 32 vCPU ≫ 8).
- **Volume**: 30 GB at `/workspace` (persists across stop/start; checkpoints +
  results live here).
- **Template**: "RunPod PyTorch".
- After it boots, copy the **direct TCP** SSH line from the Connect tab
  (`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`). The ip:port **changes on
  every restart** — always re-copy it.

---

## 2. Get the repo onto the pod

**Option A — clone from GitHub** (it's your repo; you're authed):
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
cd /workspace
git clone https://github.com/BABYZ3US/mbrl-apparatus.git mbrl && cd mbrl
```
(Private repo → use a fine-grained PAT: `git clone https://<TOKEN>@github.com/...`)

**Option B — rsync from your Mac** (no GitHub auth needed; pushes local edits):
```bash
rsync -az --delete -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
  --exclude .venv --exclude outputs --exclude checkpoints --exclude results \
  --exclude wandb --exclude __pycache__ --exclude '*.pyc' --exclude .git \
  ~/Claude/Projects/math/mbrl/ root@<ip>:/workspace/mbrl/
```

Persist the W&B key on the **volume** so it survives container restarts:
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 \
  'umask 077 && printf "%s" "<YOUR_WANDB_KEY>" > /workspace/mbrl/.wandb_key'
```

---

## 3. Bootstrap the environment

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
cd /workspace/mbrl
WANDB_API_KEY=$(cat .wandb_key) bash scripts/runpod_setup.sh
```
`runpod_setup.sh` does, in order: install uv → `uv sync` the locked env →
**CUDA check** (overlays a cu124 torch wheel if the lock pulled CPU-only wheels)
→ install `gymnasium[mujoco]` → `wandb login` → **the smoke gate** (a <2 min
Pendulum run — the rule is *never spend GPU before it passes*). It prints
`setup complete` when done. Confirm CUDA is real:
```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Launch a campaign

Run **detached** so it survives your SSH disconnect (tmux is unreliable on
restarted containers — use `setsid nohup`). Create the log dir first (the `>`
redirect opens before the script's own mkdir):

```bash
cd /workspace/mbrl && mkdir -p results/gridlogs
STACKS="spectral champion" DEPTH=4 STEPS=250000 SEEDS="0 1 2" JOBS=4 \
  nohup setsid bash scripts/run_dgate_campaign.sh \
  > results/gridlogs/_runner.log 2>&1 < /dev/null &
```

Runner knobs (all the campaign scripts share this style):

| env var | meaning | default |
|---|---|---|
| `STACKS` | which reward stacks: `mlp spectral champion` | all three |
| `SEEDS` | seed list, space-separated | `0 1` |
| `STEPS` | `training.total_env_steps` per arm | `250000` |
| `DEPTH` / `HIDDEN` | net capacity overrides (24 GB VRAM = headroom) | config default |
| `JOBS` | concurrent arms (≈4 per GPU; CPU-bound) | `4 × NGPU` |
| `NGPU` | GPUs to round-robin across (auto-detected) | `nvidia-smi -L` count |
| `PIN_GPU` | force all arms onto one GPU (add a batch beside a live run) | unset |

Other runners: `run_ensemble_grid.sh` (pessimism grid), `run_campaign2.sh`
(spectral bridge). Each arm gets a distinct `experiment.name` → its own W&B
group + local mirror; all arms are checkpoint-resumable.

---

## 5. Monitor

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 'cd /workspace/mbrl && \
  echo "procs: $(pgrep -fc "[s]cripts/train.py")"; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; \
  tail -1 results/gridlogs/_runner.log; \
  for f in results/gridlogs/dg-*.log; do \
    echo "$(basename $f .log): $(grep -c Traceback $f)TB $(grep -oE "env_steps=[0-9]+" $f|tail -1)"; done'
```
Or just watch the live charts on wandb.ai. The runner prints
`dgate campaign done: N/M succeeded` to `_runner.log` when finished.

---

## 6. Pull the verdict

**6a — on the pod (local mirrors):**
```bash
.venv/bin/python scripts/campaign_verdict.py "dg-*"     # glob your run's groups
```

**6b — from W&B (works even if the pod is gone)** — per-(stack,gate) final
eval/return + gate trajectory:
```bash
WANDB_API_KEY=$(cat .wandb_key) python - <<'PY'
import math, wandb
api = wandb.Api(timeout=60)
runs = [r for r in api.runs("pandelak-boston-college/mbrl-curvature")
        if r.group and r.group.startswith("dg-")]
latest = {}                       # dedup re-used group names -> newest run wins
for r in runs:
    if r.name not in latest or str(r.created_at) > str(latest[r.name].created_at):
        latest[r.name] = r
G = {}
for r in latest.values():
    _, stack, gate = r.group.split("-")[:3]
    ev = [x["eval/return"] for x in r.history(keys=["eval/return"], pandas=False)
          if x.get("eval/return") is not None]
    if ev: G.setdefault((stack, gate), []).append(sum(ev[-3:]) / len(ev[-3:]))
for k in sorted(G):
    xs = G[k]; m = sum(xs)/len(xs)
    s = math.sqrt(sum((x-m)**2 for x in xs)/len(xs)) if len(xs) > 1 else 0
    print(f"{k[0]:9} {k[1]:4} {m:.0f}±{s:.0f} (n={len(xs)})")
PY
```

---

## 7. Stop the pod

**Stop it from the RunPod console** (Stop, not Terminate — Stop keeps the
volume; Terminate wipes it). Billing stops on Stop.

Auto-stop-on-done from inside the pod needs a RunPod **account API key**
(the pod's embedded key is `Unauthorized` for pod control). If you set that up:
```bash
runpodctl config --apiKey <ACCOUNT_API_KEY>
( until grep -q "campaign done:" results/gridlogs/_runner.log; do sleep 120; done; \
  runpodctl stop pod "$RUNPOD_POD_ID" ) &        # detached watcher
```

---

## 8. Drive the Studio against the pod (live remote control)

The **MBRL Studio** (Godot GUI, `../godot_studio/`) runs **locally on your Mac** —
it needs a display, the pod does not have one. It talks to the apparatus over ONE TCP
socket served by `scripts/studio_bridge_server.py`: it pulls runs/metrics/sweeps for the
viz panels AND, on **submit.spec / submit.sweep**, launches `scripts/train.py` **on the
pod's GPU**. So the loop becomes: author a model graph locally → submit → it trains on
RunPod → the Run Monitor / Plots / Ablation panels stream the result back. No W&B round-trip
needed for the live view (W&B is still the durable rendezvous per the golden rule).

**Security:** the bridge server is **unauthenticated** and submit launches arbitrary
training on its host. NEVER bind it to RunPod's public IP. Keep it on `127.0.0.1` (the
default) and reach it over an **SSH local-forward** — the same key/host you already use.

**On the pod** — start the server bound to localhost (from `/workspace/mbrl`):
```bash
.venv/bin/python scripts/studio_bridge_server.py --host 127.0.0.1 --port 9009
# add --dry-run first to verify the wire without spawning real training;
# add --strict to reject specs that trip the spectral house rules.
```

**On your Mac** — forward the pod's bridge port to localhost, then launch the studio
pointed at it. The helper defaults to the `~/.ssh/runpod` key (override with `-i`) and the
same `root@<ip> -p <port>` line from the Connect tab (§1):
```bash
cd ~/Claude/Projects/math/godot_studio
./tools/tunnel_runpod.sh root@<pod-ip> -p <ssh-port> --launch   # tunnel + studio, one command
# — or in two terminals —
./tools/tunnel_runpod.sh root@<pod-ip> -p <ssh-port>            # terminal 1: holds the tunnel
./run_studio.sh --no-bridge -- --port 9009                     # terminal 2: dials 127.0.0.1:9009
```
`--no-bridge` is **essential**: the bridge runs **on the pod**, so the local studio must NOT
start its own. (Without it, `spine-studio` starts a local bridge on 9009 — colliding with the
tunnel and training on your Mac instead of the pod.) Everything **after `--`** is a Godot
launch-time backend override: `--port 9009` upserts+selects a `launch` profile (visible in the
status-bar backend picker) dialing 127.0.0.1:9009 = the tunnel. Use `-- --apparatus
127.0.0.1:9010` / `--local 9010` if 9009 is taken locally; `-- --host <ip> --port <p>` for a
non-tunnel address. The pod's ip:port changes on every restart — re-copy it each time.

---

## Gotchas (learned the hard way)

- **Container restarts wipe `~/.ssh` and `~/.netrc`** (only `/workspace`
  persists). The account-level SSH key (§0) and the volume `.wandb_key` (§2)
  are the fixes; the runners auto-source `.wandb_key`.
- **ip:port changes on every restart** — re-copy from Connect.
- **Host key changed for the same ip:port = it's a DIFFERENT machine** (RunPod
  reassigned it). Do NOT blindly `ssh-keygen -R` and trust the new key — confirm
  it's your pod first; you could be connecting to another tenant.
- **uv venvs ship no `pip`** → use `uv pip install ...`, not `.venv/bin/pip`.
- **The lock may pin CPU torch** → `runpod_setup.sh` overlays cu124; always
  confirm `torch.cuda.is_available()` before a real run.
- **`pkill -f "scripts/train.py"` matches its own SSH command** → use a bracket
  to exclude self: `pkill -f "[s]cripts/train.py"`.
- **`mkdir -p results/gridlogs` before** any `> results/gridlogs/...` redirect on
  a fresh box, or the launch silently no-ops.
- **Don't raise `JOBS` past ~4/GPU** — the GPU is already saturated; more just
  thrashes. Scale with more GPUs, not more arms per GPU.
