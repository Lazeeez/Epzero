#!/usr/bin/env bash
# Deploys (or upgrades) Epzero onto the nodes created by deploy/oracle (terraform apply):
#   - builds epzero-node natively on every node from the committed source (ARM64, NEON kernels)
#   - cross-compiles the Go router for linux/arm64 and runs it on node 1 behind Caddy (HTTPS)
#   - installs systemd units that restart crashed processes (they recover from their WAL)
#   - restarts nodes one at a time so the cluster stays available during upgrades
#
#   deploy/deploy.sh            # deploy HEAD
#   EPZERO_ORIGINS=https://example.com deploy/deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/deploy/oracle"
KEY="${EPZERO_SSH_KEY:-$HOME/.ssh/epzero_oci}"
TOKEN_FILE="${EPZERO_TOKEN_FILE:-$HOME/.epzero/token}"
ORIGINS="${EPZERO_ORIGINS:-https://lazeeez.github.io,http://localhost:5173,http://localhost:4173}"
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
git -C "$ROOT" archive --format=tar.gz -o "$SRC/epzero-src.tar.gz" HEAD engine proto deploy/systemd
(cd "$ROOT/cluster" && GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$SRC/epzero-router" ./cmd/epzero-router)

build_node() {
  local i=$1 ip=${PUBLIC[$1]}
  to "$ip" "$SRC/epzero-src.tar.gz" "$TOKEN_FILE"
  on "$ip" sudo bash -s -- "node$((i + 1))" "${PRIVATE[$i]}" <<'REMOTE'
set -euo pipefail
name=$1 private_ip=$2
export DEBIAN_FRONTEND=noninteractive
if ! command -v cmake >/dev/null || ! dpkg -s libgrpc++-dev >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq build-essential cmake ninja-build pkg-config libgrpc++-dev libprotobuf-dev \
    protobuf-compiler protobuf-compiler-grpc > /dev/null
fi
rm -rf /opt/epzero/src && mkdir -p /opt/epzero/src /opt/epzero/bin /etc/epzero
tar -xzf /tmp/epzero-src.tar.gz -C /opt/epzero/src
cmake -S /opt/epzero/src/engine -B /opt/epzero/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DEPZERO_BUILD_TESTS=OFF -DEPZERO_BUILD_BENCH=OFF > /dev/null
cmake --build /opt/epzero/build --target epzero_node > /dev/null
install -m 0755 /opt/epzero/build/epzero-node /opt/epzero/bin/epzero-node.new
install -m 0640 -o root -g epzero /tmp/token /etc/epzero/token
printf 'EPZERO_NAME=%s\nEPZERO_ADVERTISE=%s:7000\n' "$name" "$private_ip" > /etc/epzero/node.env
install -m 0644 /opt/epzero/src/deploy/systemd/epzero-node.service /etc/systemd/system/epzero-node.service
rm -f /tmp/token /tmp/epzero-src.tar.gz
echo "$name: epzero-node built on $(uname -m)"
REMOTE
}

echo "== building epzero-node on ${#PUBLIC[@]} nodes (in parallel)"
pids=()
for i in "${!PUBLIC[@]}"; do build_node "$i" & pids+=($!); done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "== rolling restart (one node at a time)"
for i in "${!PUBLIC[@]}"; do
  on "${PUBLIC[$i]}" sudo bash -c 'mv /opt/epzero/bin/epzero-node.new /opt/epzero/bin/epzero-node && systemctl daemon-reload &&
    systemctl enable --quiet epzero-node && systemctl restart epzero-node && sleep 2 && systemctl is-active epzero-node'
  sleep 6  # let the router notice and let replicas catch up before touching the next node
done

echo "== router + HTTPS on node1 ($HOST)"
NODES=$(for i in "${!PRIVATE[@]}"; do printf 'node%d=%s:7000,' $((i + 1)) "${PRIVATE[$i]}"; done)
to "${PUBLIC[0]}" "$SRC/epzero-router" "$ROOT/deploy/systemd/epzero-router.service"
on "${PUBLIC[0]}" sudo bash -s -- "${NODES%,}" "$ORIGINS" "$HOST" <<'REMOTE'
set -euo pipefail
nodes=$1 origins=$2 host=$3
export DEBIAN_FRONTEND=noninteractive
command -v caddy >/dev/null || { apt-get update -qq && apt-get install -y -qq caddy > /dev/null; }
install -m 0755 /tmp/epzero-router /opt/epzero/bin/epzero-router
install -m 0644 /tmp/epzero-router.service /etc/systemd/system/epzero-router.service
printf 'EPZERO_NODES=%s\nEPZERO_ORIGINS=%s\n' "$nodes" "$origins" > /etc/epzero/router.env
cat > /etc/caddy/Caddyfile <<EOF
$host {
	encode zstd gzip
	reverse_proxy 127.0.0.1:8080
}
EOF
systemctl daemon-reload
systemctl enable --quiet epzero-router caddy
systemctl restart epzero-router
systemctl reload caddy || systemctl restart caddy
rm -f /tmp/epzero-router /tmp/epzero-router.service
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
