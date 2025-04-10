import multiprocessing as mp
from multiprocessing import Process

def start_func():
    import torch
    import ibp_cuda as ibp
    ibp.start(120000, 1000)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    procs = []
    proc = Process(target=start_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    import torch
    import ibp_cuda as ibp
    from torch import nn as nn
    import time


    device = "cuda:1"

    tensor1, tensor2 = ibp.test(120000, 1000)
    m = nn.Linear(1000, 256).to(device)

    torch.cuda.set_device(device)

    #Warmup
    for i in range(10):
        tensor1_f =  m(tensor1)
        tensor2_f = m(tensor2)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(100):
        tensor2_f = m(tensor2)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Time taken2: {end - start}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(100):
        tensor1_f = m(tensor1)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Time taken: {end - start}")

    for proc in procs:
        proc.join()