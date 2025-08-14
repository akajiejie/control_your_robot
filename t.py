from multiprocessing import Process, Manager

def worker(d):
    import numpy as np
    manager = Manager()
    arr = np.arange(10)
    d["arr"] = arr
    d["test"] = manager.dict()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")

    manager = Manager()
    shared = manager.dict()
    for i in range(10):
        p = Process(target=worker, args=(shared,))
        p.start()
        p.join()
        p.close()
        print(shared)
        a  = dict(shared)
