#!/bin/bash
cd ~/dflash_bench
CUDA=~/miniconda3/envs/build
export LD_LIBRARY_PATH=$CUDA/lib:$HOME/dflash_bench/llama.cpp/build/bin:/usr/lib/x86_64-linux-gnu
SRV=$HOME/dflash_bench/llama.cpp/build/bin/llama-server; PY=~/dflash_bench/hfenv/bin/python
PORT=8901; N=384; mkdir -p campmtp2
T=models_mtp/Qwen3.6-27B-IQ4_XS-mtp.gguf
send_all(){ for i in $(seq 0 39); do
  $PY -c "import json;p=json.load(open('prompts40.json'))[$i][1];json.dump({'messages':[{'role':'user','content':p}],'n_predict':$N,'temperature':0,'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False},'cache_prompt':False},open('/tmp/qm2.json','w'))"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' --data @/tmp/qm2.json > campmtp2/${1}_p${i}.json 2>/dev/null; done; }
run_cfg(){ pkill -9 -f llama-server 2>/dev/null; sleep 2
  if [ -z "$2" ]; then CUDA_VISIBLE_DEVICES=0,1 $SRV -m $T -ngl 99 -c 4096 --port $PORT --host 127.0.0.1 -fa on --jinja --spec-type none > campmtp2/srv_$1.log 2>&1 &
  else CUDA_VISIBLE_DEVICES=0,1 $SRV -m $T -ngl 99 -c 4096 --port $PORT --host 127.0.0.1 -fa on --jinja --spec-type draft-mtp --spec-draft-n-max $2 > campmtp2/srv_$1.log 2>&1 &
  fi
  for i in $(seq 1 120); do curl -s http://127.0.0.1:$PORT/health 2>/dev/null|grep -q ok && break; sleep 2; done; sleep 1
  send_all "$1"; pkill -9 -f llama-server 2>/dev/null; sleep 2; echo "  done $1 $(date +%T)"; }
for C in $(echo "baseline n1 n2 n3 n4 n6" | tr ' ' '\n' | shuf); do
  if [ "$C" = baseline ]; then run_cfg mtp_baseline ""; else run_cfg mtp_$C ${C#n}; fi
done
echo MTP2_DONE $(date +%T)
