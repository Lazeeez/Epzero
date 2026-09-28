#!/usr/bin/env bash
# Oracle's Always Free Ampere pool is often "Out of host capacity"; capacity frees up as other
# tenants release instances. Retries `terraform apply` until every node exists (nodes that were
# created are kept between attempts), or until the time budget runs out.
#
#   TF_VAR_tenancy_ocid=... deploy/oracle/retry-apply.sh [interval_seconds] [max_hours]
set -uo pipefail
cd "$(dirname "$0")"
interval=${1:-180}
deadline=$(( $(date +%s) + ${2:-12} * 3600 ))
want=$(terraform output -json 2>/dev/null >/dev/null; grep -Eo 'default *= *[0-9]+' variables.tf | head -1 | grep -Eo '[0-9]+')

attempt=0
while :; do
  attempt=$((attempt + 1))
  if terraform apply -auto-approve -input=false -no-color > /tmp/oryx-apply.log 2>&1; then
    echo "$(date -u +%FT%TZ) attempt $attempt: all nodes created"
    terraform output -no-color
    exit 0
  fi
  have=$(terraform state list 2>/dev/null | grep -c 'oci_core_instance.node')
  reason=$(grep -Eo 'Out of host capacity|LimitExceeded|NotAuthorized[A-Za-z]*|InternalError' /tmp/oryx-apply.log | sort -u | tr '\n' ' ')
  echo "$(date -u +%FT%TZ) attempt $attempt: $have/${want:-3} nodes, ${reason:-see /tmp/oryx-apply.log}"
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "giving up after $attempt attempts"
    exit 1
  fi
  sleep "$interval"
done
