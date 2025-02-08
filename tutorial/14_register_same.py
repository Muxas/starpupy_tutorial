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
    a_h = starpu.Handle(a)
    # Registering the same data with another handle is an error
    a2_h = starpu.Handle(a)
    b_h = starpu.Handle(b)
    c_h = starpu.Handle(c)
    # Call functions as before
    fut1 = axpy(1.0, a_h, b_h)
    fut2 = axpy(-1.0, c_h, b_h)
    print("Value of b after submitting tasks but before they finish is", b)
    # Now acquire and release are methods of the Handle. Do not use
    # starpu.acquire and starpu.release in this case
    a2_h.acquire(mode="R")
    print("Print value a before tasks are executed", a)
    a2_h.release()
    b_h.acquire(mode="R")
    print("The result of function (value b) is", b)
    b_h.release()
    await fut1, fut2


if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
