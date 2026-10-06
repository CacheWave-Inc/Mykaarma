cd /home/cw-biz1/vjepa-bench
RES=320 LRB=1e-5 OUTSUF=_expA .venv/bin/python train_det.py battery 4 3000 bare > exp_A.log 2>&1
RES=320 LRB=1e-4 OUTSUF=_expB .venv/bin/python train_det.py battery 4 3000 bare > exp_B.log 2>&1
RES=448 LRB=1e-4 OUTSUF=_expC .venv/bin/python train_det.py battery 4 3000 bare > exp_C.log 2>&1
RES=320 LRB=1e-4 AUGB=0.9,1.1 OUTSUF=_expD .venv/bin/python train_det.py battery 4 3000 bare > exp_D.log 2>&1
echo done > exp_done.flag
