import asyncio
import time

import numpy as np
import starpu


@starpu.delayed(name="step1", perfmodel="step1")
# Describe access modifiers to provided inputs
# It is possible to change values inplace
@starpu.access(x="RW", t="R")
def step1(x, t):
    time.sleep(t)
    x *= 2


@starpu.delayed(name="step2", perfmodel="step2")
# Describe access modifiers to provided inputs
# It is possible to change values inplace
@starpu.access(x="RW")
def step2(x, t):
    time.sleep(t)
    x += 2


async def main():
    a = np.random.randn(10)
    a_h = starpu.Handle(a)
    b = np.array([1.0, 1.0])
    c = np.array([2.0, 2.0, 2.0])
    d = np.array([4.0, 4.0, 4.0, 4.0])
    # Measure time
    t0 = time.time()
    # Call step1
    step1(a_h, 3) # step1(0) takes 3 seconds
    step1(b, 1) # step1(1) takes 1 second
    step1(c, 1) # step1(2) takes 1 second
    step1(d, 6) # step1(3) takes 6 seconds
    # Call step2
    step2(a_h, 1) # step2(0) takes 1 second
    step2(b, 6) # step2(1) takes 6 seconds
    step2(c, 4) # step2(2) takes 4 seconds
    step2(d, 1) # step2(3) takes 1 second
    starpu.starpupy.task_wait_for_all()
    # Measure time
    t0 = time.time() - t0
    print(f"Execution time is {t0} seconds")


if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
