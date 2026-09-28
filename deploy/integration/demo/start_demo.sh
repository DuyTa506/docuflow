#!/usr/bin/env bash
# Demo tích hợp DocuFlow ↔ hệ thống thư viện (SQL Server).
#
#   bash deploy/integration/demo/start_demo.sh
#
# 1. SQL Server (container) đóng vai CSDL thư viện: LibraryDB.
# 2. Cấp API key cho "Thư viện Demo" (lưu ở .state/, không commit).
# 3. Chạy một API DocuFlow phụ ở :8023 và một worker tích hợp riêng trên hàng
#    đợi docuflow-integration-demo, ghi kết quả vào LibraryDB. OCR, tổng thuật,
#    dịch vẫn do các worker DocuFlow đang chạy đảm nhận.
# 4. library_system.py run: gửi tài liệu, nhận callback, in kết quả trong MSSQL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STATE="$HERE/.state"
mkdir -p "$STATE"
PY="$ROOT/.venv/bin/python"
SA_PW="${LIBRARY_SA_PASSWORD:-DocuFlow#Demo2026}"
MSSQL_PORT="${LIBRARY_MSSQL_PORT:-14330}"
API_PORT="${DEMO_API_PORT:-8023}"

docker compose -f "$HERE/docker-compose.yml" up -d
until [ "$(docker inspect -f '{{.State.Health.Status}}' docuflow-library-mssql)" = healthy ]; do sleep 3; done

if [ ! -s "$STATE/client.json" ]; then
    (cd "$ROOT" && "$PY" scripts/integration_client.py create "Thư viện Demo" \
        --callback-url "http://localhost:${CALLBACK_PORT:-9099}/docuflow/callback" --json) \
        | tail -1 > "$STATE/client.json"
fi

export DOCUFLOW_URL="http://localhost:$API_PORT"
export DOCUFLOW_API_KEY="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['api_key'])" "$STATE/client.json")"
export DOCUFLOW_WEBHOOK_SECRET="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['webhook_secret'])" "$STATE/client.json")"
export LIBRARY_MSSQL_URL="mssql+pymssql://sa:${SA_PW}@localhost:${MSSQL_PORT}/master"
export TEMPORAL_INTEGRATION_TASK_QUEUE=docuflow-integration-demo
export INTEGRATION_PUBLIC_BASE_URL="$DOCUFLOW_URL"
export INTEGRATION_POLL_SECONDS=10

cd "$ROOT"
"$PY" -m uvicorn serving.workflow_api:app --host 127.0.0.1 --port "$API_PORT" > "$STATE/api.log" 2>&1 &
API_PID=$!
INTEGRATION_MSSQL_URL="mssql+pymssql://sa:${SA_PW}@localhost:${MSSQL_PORT}/LibraryDB" \
    "$PY" "$HERE/demo_worker.py" > "$STATE/worker.log" 2>&1 &
WORKER_PID=$!
trap 'kill $API_PID $WORKER_PID 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$API_PORT/api" > /dev/null; do sleep 2; done

"$PY" "$HERE/library_system.py" run
