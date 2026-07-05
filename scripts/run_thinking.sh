#!/bin/bash
cd ~/dflash_bench
CUDA=~/miniconda3/envs/build
export LD_LIBRARY_PATH=$CUDA/lib:$HOME/dflash_bench/llama.cpp/build/bin:/usr/lib/x86_64-linux-gnu
SRV=$HOME/dflash_bench/llama.cpp/build/bin/llama-server; PY=~/dflash_bench/hfenv/bin/python
PORT=8901; N=384; mkdir -p camp3t
DENSE_T=models/Qwen3.6-27B-IQ4_XS.gguf; DENSE_D=models/Qwen3.6-27B-DFlash-Q8_0.gguf
MOE_T=models_moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf; MOE_D=models_moe/Qwen3.6-35B-A3B-DFlash-q8_0.gguf
send_all(){ for i in $(seq 0 39); do
  $PY -c "import json;p=json.load(open('prompts40.json'))[$i][1];json.dump({'messages':[{'role':'user','content':p}],'n_predict':$N,'temperature':0,'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':True},'cache_prompt':False},open('/tmp/q3t.json','w'))"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' --data @/tmp/q3t.json > camp3t/${1}_p${i}.json 2>/dev/null; done; }
run_cfg(){ pkill -9 -f llama-server 2>/dev/null; sleep 2
  if [ -z "$3" ]; then CUDA_VISIBLE_DEVICES=0,1 $SRV -m "$2" -ngl 99 -c 4096 --port $PORT --host 127.0.0.1 -fa on --jinja --spec-type none > camp3t/srv_$1.log 2>&1 &
  else CUDA_VISIBLE_DEVICES=0,1 $SRV -m "$2" -md "$3" -ngl 99 -ngld 99 -c 4096 --port $PORT --host 127.0.0.1 -fa on --jinja --spec-type draft-dflash --spec-draft-n-max $4 > camp3t/srv_$1.log 2>&1 &
  fi
  for i in $(seq 1 130); do curl -s http://127.0.0.1:$PORT/health 2>/dev/null|grep -q ok && break; sleep 2; done; sleep 1
  send_all "$1"; pkill -9 -f llama-server 2>/dev/null; sleep 2; echo "  done $1 $(date +%T)"; }
for M in dense moe; do
  if [ "$M" = dense ]; then T=$DENSE_T; D=$DENSE_D; else T=$MOE_T; D=$MOE_D; fi
  for C in $(echo "baseline n1 n3 n6 n8 n15" | tr ' ' '\n' | shuf); do
    if [ "$C" = baseline ]; then run_cfg "${M}_baseline" "$T" "" ""; else run_cfg "${M}_${C}" "$T" "$D" "${C#n}"; fi
  done
done
echo P3_DONE $(date +%T)
