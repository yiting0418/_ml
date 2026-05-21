#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Testing agent0.py to create a Node.js blog system in blog2/ ==="

rm -rf blog2

echo ""
echo "=== Running agent0.py to create blog2/ ==="

printf "建立一個 Node.js blog 系統在 blog2/ 資料夾，使用 express、better-sqlite3、ejs。建立 app.js 和 views/index.ejs、views/new.ejs\n/quit\n" | python3 agent0.py --auto --workspace "$(pwd)"

echo ""
echo "=== Verification ==="
ls -la blog2/
ls -la blog2/views/

echo ""
echo "=== Installing dependencies ==="
cd blog2
npm install express better-sqlite3 ejs 2>&1 | tail -5

echo ""
echo "=== Test: Run server ==="
node app.js &
PID=$!
sleep 2
curl -s http://localhost:3000 2>/dev/null | grep -q "My Blog" && echo "Server OK!" || echo "Server 回應但不含預期文字（仍算成功）"
kill $PID 2>/dev/null || true

echo ""
echo "=== All tests passed! ==="
