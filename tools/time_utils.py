import time

def tik(): 
    return time.time()

def tok(st_time, ms=True):
    end_time = time.time() - st_time
    if ms: end_time *= 1000
    return int(end_time)

