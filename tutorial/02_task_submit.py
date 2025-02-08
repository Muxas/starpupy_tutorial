import asyncio

import starpu


# Simple Python function
def add(a, b):
    return a + b


async def main():
    fut = starpu.task_submit()(add, 1, 2)
    res = await fut
    print("The result of function is", res)


if __name__ == "__main__":
    # Init StarPU before doint any computations with its help
    starpu.init()
    # Do some work here
    asyncio.run(main())
    # Shut down StarPU explicitly in the end of working with it
    starpu.shutdown()
