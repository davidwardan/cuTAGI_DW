import multiprocessing as mp
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def run_experiments_parallel(
    tasks: List[T],
    worker_fn: Callable[..., None],
    worker_args_mapper: Callable[[T, str], Tuple],
    device_ids: Sequence[str] = ("cuda:0", "cuda:1"),
) -> None:
    """
    Run a list of tasks in parallel using a pool of devices.

    Args:
        tasks: List of task items (e.g. tuples of seed, experiment).
        worker_fn: The function to run in the subprocess.
        worker_args_mapper: A function that takes a task item and a device string,
                            and returns the tuple of arguments to pass to worker_fn.
        device_ids: List of device strings to manage.
    """
    ctx = mp.get_context("spawn")

    # We'll manage a list of active processes and available devices
    active: List[Tuple[mp.Process, str]] = []
    available_devices: List[str] = list(device_ids)

    # Work queue
    pending_tasks = list(tasks)

    while pending_tasks or active:
        # Launch as many jobs as we have free devices
        while pending_tasks and available_devices:
            task = pending_tasks.pop(0)
            device = available_devices.pop(0)

            args = worker_args_mapper(task, device)

            # We assume the worker function prints its own start message if desired,
            # or we could print a generic one here.
            # For now, let's leave printing to the caller or the worker if specific details are needed,
            # but the original code printed: "Launching ... on device ..."
            # We can't easily reconstruct that exact string without more info,
            # so we'll let the worker handle logging or just proceed silently here.

            p = ctx.Process(target=worker_fn, args=args)
            p.start()
            active.append((p, device))

        # If nothing running, continue (should only happen at the very beginning if no tasks)
        if not active:
            continue

        # Wait for the first active process to complete
        # Note: This simple logic waits for the *first launched* process in the active list.
        # A better approach might be to check all active processes, but the original code
        # did `proc, dev = active.pop(0); proc.join()`. This effectively enforces a FIFO
        # completion wait on the *slots*, which might be suboptimal if a later task finishes
        # earlier, but it's safe and simple. We'll stick to the original logic to minimize risk.
        proc, dev = active.pop(0)
        proc.join()

        if proc.exitcode:
            raise RuntimeError(
                f"Experiment on {dev} failed with exit code {proc.exitcode}"
            )

        # Free up the device
        available_devices.append(dev)
