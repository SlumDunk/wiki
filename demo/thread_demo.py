import threading
import time
from queue import Queue


def thread_job():
    """
    线程的工作内容
    :return:
    """
    print('T1 start\n')
    for i in range(10):
        time.sleep(0.1)
    print('T1 finish\n')
    print('this is an child thread, the number is %s' % threading.current_thread())


pass


def thread_2_job():
    """
    线程2的工作
    :return:
    """
    print('T2 start\n')
    print('T2 end\n')


pass


def job(l, queue):
    """
    每个线程的任务
    :param l:
    :param queue:
    :return:
    """
    for i in range(len(l)):
        l[i] = l[i] ** 2
    queue.put(l)


# 加队列
def multi_thread():
    """
    启动多线程
    :return:
    """
    q = Queue()
    threads = []
    data = [[1, 2, 3], [3, 4, 5], [4, 4, 4], [5, 5, 5]]
    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q), name='i')
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    results = []

    for _ in range(4):
        results.append(q.get())


pass


# 锁
def job1():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 1
        print('job 1', A)
    lock.release()


pass


def job2():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 10
        print('job 2', A)
    lock.release()


pass


def main():
    added_thread = threading.Thread(target=thread_job, name='T1')
    added_thread.start()
    thread2 = threading.Thread(target=thread_2_job, name='T2')
    # 等待子线程运行完
    added_thread.join()
    thread2.join()
    print('all done\n')
    print(threading.active_count())
    print(threading.enumerate())
    print(threading.current_thread())


pass

if __name__ == '__main__':
    main()

    A = 0
    lock = threading.Lock()
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
