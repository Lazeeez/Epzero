#!/usr/bin/env bash
# Deploys (or upgrades) Oryx onto the nodes created by deploy/oracle (terraform apply):
#   - builds oryx-node natively on every node from the committed source (ARM64, NEON kernels)
#   - cross-compiles the Go router for linux/arm64 and runs it on node 1 behind Caddy (HTTPS)
#   - installs systemd units that restart crashed processes (they recover from their WAL)
#   - restarts nodes one at a time so the cluster stays available during upgrades
#
#   deploy/deploy.sh            # deploy HEAD
#   ORYX_ORIGINS=https://example.com deploy/deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/deploy/oracle"
KEY="${ORYX_SSH_KEY:-$HOME/.ssh/oryx_oci}"
TOKEN_FILE="${ORYX_TOKEN_FILE:-$HOME/.oryx/token}"
ORIGINS="${ORYX_ORIGINS:-https://lazeeez.github.io,http://localhost:5173,http://localhost:4173}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15)

json() { terraform -chdir="$TF_DIR" output -json "$1" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'; }
read -r -a PUBLIC <<< "$(json public_ips)"
read -r -a PRIVATE <<< "$(json private_ips)"
ROUTER_URL="$(terraform -chdir="$TF_DIR" output -raw router_url)"
HOST="${ROUTER_URL#https://}"
[ "${#PUBLIC[@]}" -ge 2 ] || { echo "need at least 2 nodes; run terraform apply first" >&2; exit 1; }

on() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "$@"; }
to() { scp -q "${SSH_OPTS[@]}" "${@:2}" "ubuntu@$1:/tmp/"; }

# Admin token for writes and chaos: generated once, kept outside the repository.
if [ ! -s "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  (umask 077; openssl rand -hex 24 > "$TOKEN_FILE")
fi

echo "== packaging source $(git -C "$ROOT" rev-parse --short HEAD)"
SRC=$(mktemp -d)
git -C "$ROOT" archive --format=tar.gz -o "$SRC/oryx-src.tar.gz" HEAD engine proto deploy/systemd
(cd "$ROOT/cluster" && GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$SRC/oryx-router" ./cmd/oryx-router)

build_node() {
  local i=$1 ip=${PUBLIC[$1]}
  to "$ip" "$SRC/oryx-src.tar.gz" "$TOKEN_FILE"
  on "$ip" sudo bash -s -- "node$((i + 1))" "${PRIVATE[$i]}" <<'REMOTE'
set -euo pipefail
name=$1 private_ip=$2
export DEBIAN_FRONTEND=noninteractive
if ! command -v cmake >/dev/null || ! dpkg -s libgrpc++-dev >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq build-essential cmake ninja-build pkg-config libgrpc++-dev libprotobuf-dev \
    protobuf-compiler protobuf-compiler-grpc > /dev/null
fi
rm -rf /opt/oryx/src && mkdir -p /opt/oryx/src /opt/oryx/bin /etc/oryx
tar -xzf /tmp/oryx-src.tar.gz -C /opt/oryx/src
cmake -S /opt/oryx/src/engine -B /opt/oryx/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DORYX_BUILD_TESTS=OFF -DORYX_BUILD_BENCH=OFF > /dev/null
cmake --build /opt/oryx/build --target oryx_node > /dev/null
install -m 0755 /opt/oryx/build/oryx-node /opt/oryx/bin/oryx-node.new
install -m 0640 -o root -g oryx /tmp/token /etc/oryx/token
printf 'ORYX_NAME=%s\nORYX_ADVERTISE=%s:7000\n' "$name" "$private_ip" > /etc/oryx/node.env
install -m 0644 /opt/oryx/src/deploy/systemd/oryx-node.service /etc/systemd/system/oryx-node.service
rm -f /tmp/token /tmp/oryx-src.tar.gz
echo "$name: oryx-node built on $(uname -m)"
REMOTE
}

echo "== building oryx-node on ${#PUBLIC[@]} nodes (in parallel)"
pids=()
for i in "${!PUBLIC[@]}"; do build_node "$i" & pids+=($!); done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "== rolling restart (one node at a time)"
for i in "${!PUBLIC[@]}"; do
  on "${PUBLIC[$i]}" sudo bash -c 'mv /opt/oryx/bin/oryx-node.new /opt/oryx/bin/oryx-node && systemctl daemon-reload &&
    systemctl enable --quiet oryx-node && systemctl restart oryx-node && sleep 2 && systemctl is-active oryx-node'
  sleep 6  # let the router notice and let replicas catch up before touching the next node
done

echo "== router + HTTPS on node1 ($HOST)"
NODES=$(for i in "${!PRIVATE[@]}"; do printf 'node%d=%s:7000,' $((i + 1)) "${PRIVATE[$i]}"; done)
to "${PUBLIC[0]}" "$SRC/oryx-router" "$ROOT/deploy/systemd/oryx-router.service"
on "${PUBLIC[0]}" sudo bash -s -- "${NODES%,}" "$ORIGINS" "$HOST" <<'REMOTE'
set -euo pipefail
nodes=$1 origins=$2 host=$3
export DEBIAN_FRONTEND=noninteractive
command -v caddy >/dev/null || { apt-get update -qq && apt-get install -y -qq caddy > /dev/null; }
install -m 0755 /tmp/oryx-router /opt/oryx/bin/oryx-router
install -m 0644 /tmp/oryx-router.service /etc/systemd/system/oryx-router.service
printf 'ORYX_NODES=%s\nORYX_ORIGINS=%s\n' "$nodes" "$origins" > /etc/oryx/router.env
cat > /etc/caddy/Caddyfile <<EOF
$host {
	encode zstd gzip
	reverse_proxy 127.0.0.1:8080
}
EOF
systemctl daemon-reload
systemctl enable --quiet oryx-router caddy
systemctl restart oryx-router
systemctl reload caddy || systemctl restart caddy
rm -f /tmp/oryx-router /tmp/oryx-router.service
REMOTE
rm -rf "$SRC"

echo "== waiting for $ROUTER_URL"
for _ in $(seq 1 60); do
  if curl -fsS "$ROUTER_URL/healthz" > /dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS "$ROUTER_URL/v1/cluster" | python3 -c '
import json, sys
v = json.load(sys.stdin)
for n in v["nodes"]: print(f"  {n[\"name\"]}: {\"up\" if n[\"up\"] else \"DOWN\"}  simd={n[\"simd\"]}")
for s in v["shards"]: print(f"  shard {s[\"id\"]}: primary={s[\"primary\"]} replica={s[\"replica\"]} epoch={s[\"epoch\"]} in_sync={s[\"replica_in_sync\"]} vectors={s[\"vectors\"]}")'
echo "router: $ROUTER_URL   (token: $TOKEN_FILE)"
