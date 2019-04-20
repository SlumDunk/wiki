import multiprocessing as mp
import threading as td
import time


def job(x):
    return x * x


pass


def batch_job(queue):
    res = 0
    for i in range(1000):
        res += i + i ** 2 + i ** 3
    queue.put(res)
    pass


# 进程池
def multi_pool():
    pool = mp.Pool(processes=2)
    res = pool.map(job, range(10))
    print(res)
    res = pool.apply_async(job, (2,))
    print(res.get())
    multi_res = [pool.apply_async(job, (i,)) for i in range(10)]
    print([res.get() for res in multi_res])


# 队列
def multi_core():
    q = mp.Queue()
    p2 = mp.Process(target=batch_job, args=(q,))
    p3 = mp.Process(target=batch_job, args=(q,))
    p2.start()
    p2.join()
    p3.start()
    p3.join()
    res1 = q.get()
    res2 = q.get()
    print(res1 + res2)


def job2(v, num, lock):
    lock.acquire()
    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print(v.value)
    lock.release()


# 进程锁
def multi_core_lock():
    # 共享内存
    value = mp.Value('d', 1)
    array = mp.Array('i', [1, 2, 3])
    lock = mp.Lock()
    p1 = mp.Process(target=job2, args=(value, 1, lock))
    p2 = mp.Process(target=job2, args=(value, 3, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


def hello_world():
    t1 = td.Thread(target=job, args=[2, ])
    p1 = mp.Process(target=job, args=[2, ])
    t1.start()
    p1.start()
    t1.join()
    p1.join()


if __name__ == '__main__':
    hello_world()
    multi_core()
    multi_pool()
