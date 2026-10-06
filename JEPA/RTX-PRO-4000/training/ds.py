import time, threading, torch, numpy as np
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float16).cuda().eval()
X = torch.randn(45,16,3,256,256).half().cuda()
def run(chunk, stream=None):
    ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())
    with torch.no_grad(), ctx:
        for i in range(chunk.shape[0]):
            m(chunk[i:i+1], skip_predictor=True)
def timed(fn):
    fn(); torch.cuda.synchronize()
    t=time.time(); fn(); torch.cuda.synchronize(); return time.time()-t
one = timed(lambda: run(X))
print(f"1 stream, all 45 clips: {one:.2f}s")
def split(k):
    chunks = torch.chunk(X, k)
    streams = [torch.cuda.Stream() for _ in range(k)]
    def go():
        th=[threading.Thread(target=run,args=(c,s)) for c,s in zip(chunks,streams)]
        [t.start() for t in th]; [t.join() for t in th]
    return timed(go)
for k in (2,3,4):
    print(f"data split across {k} streams on ONE GPU: {split(k):.2f}s")
half = timed(lambda: run(X[:23])); print(f"one half (23 clips) alone on the GPU: {half:.2f}s  <- what each of two nodes would take")
