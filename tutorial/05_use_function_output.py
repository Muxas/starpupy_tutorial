import asyncio

import starpu


@starpu.delayed(name="add", color=2, perfmodel="add")
# name (string, default: None) : Set the name of the task. This can be useful
#       for debugging purposes.
# synchronous (unsigned, default: 0) : If this flag is set, task_submit() only
#       returns when the task has been executed (or if no worker is able to
#       process the task). Otherwise, task_submit() returns immediately.
# priority (int, default: 0) : Set the level of priority for the task. This is
#       an integer value whose value must be greater than the return value of
#       the function starpu.sched_get_min_priority() (for the least important
#       tasks), and lower or equal to the return value of the function
#       starpu.sched_get_max_priority() (for the most important tasks). Default
#       priority is defined as 0 in order to allow static task initialization.
#       Scheduling strategies that take priorities into account can use this
#       parameter to take better scheduling decisions, but the scheduling
#       policy may also ignore it.
# color (unsigned, default: None) : Set the color of the task to be used in
#       dag.dot.
# flops (double, default: None) : Set the number of floating points operations
#       that the task will have to achieve. This is useful for easily getting
#       GFlops/s curves from the function starpu.perfmodel_plot, and for the
#       hypervisor load balancing.
# perfmodel (string, default: None) : Set the name of the performance model.
#       This name will be used as the filename where the performance model
#       information will be saved. After the task is executed, one can call
#       the function starpu.perfmodel_plot() by giving the symbol of perfmodel
#       to view its performance curve.
def add(a, b):
    return a + b


@starpu.delayed(name="sub", perfmodel="sub")
def sub(a, b):
    return a - b


async def main():
    # Call delayed function one after another, using output as input
    fut1 = add(1, 2)
    fut2 = sub(fut1, 3)
    res = await fut2
    print("The result of function is", res)

if __name__ == "__main__":
    starpu.init()
    asyncio.run(main())
    starpu.shutdown()
