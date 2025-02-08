import asyncio
import time

import numpy as np
import starpu


@starpu.delayed(name="axpy", perfmodel="axpy")
def axpy(alpha, x, y):
    time.sleep(10)
    z = np.empty_like(x)
    for i in range(np.size(x)):
        z[i] = alpha*x[i] + y[i]
    return z


async def main():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    fut1 = axpy(1.0, a, b)
    fut2 = axpy(-1.0, c, fut1)
    res = await fut2
    print("The result of function is", res)

if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
