import asyncio
import time

import starpu


@starpu.delayed(name="axpy_list", perfmodel="axpy_list")
# Describe access modifiers to provided inputs
# It is possible to change values inplace
@starpu.access(y="RW", x="R")
def axpy(alpha, x, y):
    time.sleep(1)
    for i in range(len(y)):
        y[i] += alpha*x[i]


async def main():
    # Now a, b and c are Python lists instead of Numpy arrays
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    # Objects, not compatible with the buffer protocol, shall be wrapped
    a_h = starpu.Handle(a)
    b_h = starpu.Handle(b)
    c_h = starpu.Handle(c)
    # Call functions as before
    fut1 = axpy(1.0, a_h, b_h)
    fut2 = axpy(-1.0, c_h, b_h)
    print("Value of b after submitting tasks but before they finish is", b_h.get())
    # Acquiring object returns its actual value
    z = a_h.acquire(mode="R")
    print("Print value a before tasks are executed", z)
    # For some strange reason doing a_h.release() leads to an error
    starpu.release(z)
    z = b_h.acquire(mode="R")
    print("The result of function (value b) is", z)
    starpu.release(z)
    await fut1, fut2


if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
