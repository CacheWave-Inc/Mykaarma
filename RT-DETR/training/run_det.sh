for t in plate brake battery; do .venv/bin/python train_det.py $t 12 > det_$t.log 2>&1; done; echo finished > det_done.flag
