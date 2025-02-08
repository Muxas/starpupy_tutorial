import asyncio
import time

import numpy as np
import starpu


@starpu.delayed(name="axpy", perfmodel="axpy")
# Describe access modifiers to provided inputs
# It is possible to change values inplace
@starpu.access(y="RW", x="R")
def axpy(alpha, x, y):
    time.sleep(1)
    for i in range(np.size(y)):
        y[i] += alpha*x[i]


async def main():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    fut1 = axpy(1.0, a, b)
    fut2 = axpy(-1.0, c, b)
    print("Value of b after submitting tasks but before they finish is", b)
    # Instead of waiting for future fut2 we can explicitly ask for a buffer
    # starpu.acquire(b) tells StarPU that the buffer b is to be updated in the
    # main thread now. Can only be called outside tasks
    starpu.acquire(b, mode="R") # Default is "R", other options are "W" and "RW"
    print("The result of function (value b) is", b)
    # starpu.release(b) tells StarPU that we finished changing the buffer b
    # and it can be used in other tasks
    starpu.release(b)
    await fut1, fut2


if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
