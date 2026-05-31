#!/usr/bin/env bash
# End-to-end smoke test for the local CPU RAG stack.
# Assumes biltiq-embed (:8204) and biltiq-rag (:8205) are up. Generation uses
# whatever BILT_CHAT_BASE the rag service was launched with (default Qwen :8202).
set -u
EMBED=http://localhost:8204
RAG=http://localhost:8205
pass=0; fail=0
chk(){ if eval "$2"; then echo "  ✓ $1"; pass=$((pass+1)); else echo "  ✗ $1"; fail=$((fail+1)); fi; }

echo "== health =="
chk "embed up"  '[ "$(curl -s -o /dev/null -w %{http_code} $EMBED/health)" = 200 ]'
chk "rag up"    '[ "$(curl -s -o /dev/null -w %{http_code} $RAG/health)" = 200 ]'

echo "== embeddings (dim + multilingual) =="
DIM=$(curl -s $EMBED/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["नमस्ते दुनिया"],"input_type":"passage"}' \
  | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"][0]["embedding"]))')
chk "embedding dim == 384" '[ "'"$DIM"'" = 384 ]'

echo "== rerank (relevant beats irrelevant) =="
ORDER=$(curl -s $EMBED/rerank -H 'Content-Type: application/json' \
  -d '{"query":"capital of France","documents":["bananas are yellow","Paris is the capital of France"],"top_n":2}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["results"][0]["index"])')
chk "top reranked doc is the Paris one (index 1)" '[ "'"$ORDER"'" = 1 ]'

echo "== rag ingest + grounded query =="
curl -s $RAG/rag/reset >/dev/null
curl -s $RAG/rag/ingest -H 'Content-Type: application/json' -d '{
  "documents":[
    {"text":"Manthan runs native vLLM on port 8082 and explicitly does NOT use Docker.","source":"a"},
    {"text":"The reranker model is bge-reranker-v2-m3, chosen for multilingual support.","source":"b"}
  ]}' | python3 -m json.tool
ANS=$(curl -s $RAG/rag/query -H 'Content-Type: application/json' \
  -d '{"query":"What port does Manthan use and does it use Docker?","max_tokens":200}')
echo "$ANS" | python3 -m json.tool
chk "answer mentions 8082"  'echo "$ANS" | grep -q 8082'
chk "answer says no Docker" 'echo "$ANS" | grep -iqE "no(t| ).*docker|does not use docker|without docker"'

echo "== result: $pass passed, $fail failed =="
[ "$fail" = 0 ]
