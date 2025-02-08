import asyncio

import starpu


async def main():
    pass


if __name__ == "__main__":
    # Init StarPU before doint any computations with its help
    starpu.init()
    # Do some work here
    asyncio.run(main())
    # Shut down StarPU explicitly in the end of working with it
    starpu.shutdown()
