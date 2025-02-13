import time
import concurrent.futures
import os
from tqdm import tqdm


def do_something(x=0, y=1, z=2, t=1):
    print(os.getpid())
    print(f'Sleeping for {t} second')
    # print(x, y, z)
    time.sleep(t)
    print('Done sleeping')
    return str(os.getpid())

def unpack_and_call(func, kwargs):
    return func(**kwargs)


def main():
    # comment to commit
    start = time.perf_counter()
    from itertools import repeat
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     list(tqdm(executor.map(do_something, repeat(100), repeat(101), repeat(102), range(5)), total=len(range(5))))
    # with tqdm(total=5) as pbar:
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         futures = [executor.submit(do_something, 100, 101, 102, x) for x in range(5)]
    #         for future in concurrent.futures.as_completed(futures):
    #             pbar.update(1)
    #             print(future.result())

    # results = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for result in tqdm(executor.map(do_something, repeat(100), repeat(101), repeat(102), range(5)), total=5):
    #         results.append(result)

    # print(results)

    # args = range(5)
    # from functools import partial
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     func = partial(do_something, 100, 200, 300)
        # list(tqdm(executor.map(func, args), total=len(args)))

    # args = range(5)
    # from functools import partial
    # func = partial(do_something, 100, 200, 300)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     list(executor.map(func, args))

    from functools import partial
    func = partial(do_something, x=100, y=200, z=300)
    kwargs = [{"t":n} for n in range(5)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(unpack_and_call, [func]*len(kwargs), kwargs), total=len(kwargs)))

    print(results)
    end = time.perf_counter()

    print(f'End time: {(end-start):.2f} seconds')

main()
