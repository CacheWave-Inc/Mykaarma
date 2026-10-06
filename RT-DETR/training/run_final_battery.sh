cd /home/cw-biz1/vjepa-bench
RES=448 LRB=1e-4 LAST=1 .venv/bin/python train_det.py battery 10 1000000000 bare > det_battery_bare_final.log 2>&1
RES=448 LRB=1e-4 LAST=1 .venv/bin/python train_det.py battery 10 1000000000 covered > det_battery_covered_final.log 2>&1
echo finished > battery_final_done.flag
