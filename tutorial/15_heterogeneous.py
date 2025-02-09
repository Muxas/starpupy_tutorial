import asyncio
import time

import numpy as np
import starpu


@starpu.delayed(name="step1", perfmodel="step1")
# Describe access modifiers to provided inputs
# It is possible to change values inplace
@starpu.access(x="RW")
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
    a = np.array([0])
    b = np.array([1])
    c = np.array([2])
    d = np.array([4])
    # Measure time
    t0 = time.time()
    # Call step1
    step1(a, 3) # step1(0) takes 3 seconds
    step1(b, 1) # step1(1) takes 1 second
    step1(c, 1) # step1(2) takes 1 second
    step1(d, 6) # step1(3) takes 6 seconds
    # Implicit barrier of pragma omp for loop
    starpu.starpupy.task_wait_for_all()
    # Call step2
    step2(a, 1) # step2(0) takes 1 second
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
