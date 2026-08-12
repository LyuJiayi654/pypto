# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``make_tensor_arg`` used by generated distributed orchestration code.

The generated ``orchestration/host_orch.py`` builds simpler ``TaskArgs`` by
calling ``make_tensor_arg(tensors["<name>"])`` for every tensor parameter.
This pypto-owned wrapper widens that conversion to also accept worker-resident
:class:`~pypto.runtime.DeviceTensor` handles (and already-built simpler
``Tensor`` values), so distributed programs can be invoked with
pre-uploaded device buffers — mirroring the L2 path in
:func:`pypto.runtime.runner.execute_compiled`.

Host ``torch.Tensor`` arguments are named through the active simpler Worker;
device-resident arguments are resolved to runtime-owned ``Buffer`` views.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import Any

_ACTIVE_WORKER: ContextVar[Any | None] = ContextVar("_pypto_tensor_arg_worker", default=None)
_DEVICE_RESOLVER: ContextVar[Any | None] = ContextVar("_pypto_device_tensor_resolver", default=None)


@contextmanager
def bind_tensor_arg_context(worker: Any, device_resolver: Any = None):
    """Bind the Simpler Worker that owns generated orchestration arguments."""
    worker_token = _ACTIVE_WORKER.set(worker)
    resolver_token = _DEVICE_RESOLVER.set(device_resolver)
    try:
        yield
    finally:
        _DEVICE_RESOLVER.reset(resolver_token)
        _ACTIVE_WORKER.reset(worker_token)


@cache
def _modules() -> tuple[Any, Any]:
    """Import and cache the two runtime modules on first ``make_tensor_arg`` call.

    The imports stay inside this function so importing pypto never requires
    simpler (only available in the runtime environment). ``functools.cache``
    runs the body once — instead of on every call — which matters because the
    generated ``host_orch`` calls ``make_tensor_arg`` once per tensor per rank
    (~90 tensors × world_size), where per-call ``from ... import`` was pure
    overhead on the host dispatch loop.

    Only the *module objects* are cached; the individual symbols (``Tensor``,
    ``DeviceTensor``, ``make_tensor_arg``) are
    resolved via attribute access on every call. Caching the module rather than
    the bound symbols keeps ``make_tensor_arg`` responsive to test monkeypatches
    of ``task_interface.make_tensor_arg`` (see ``tests/ut/runtime``), while still
    paying the import cost only once.

    Returns:
        ``(task_interface, device_tensor)`` module objects.
    """
    from . import device_tensor, task_interface  # noqa: PLC0415

    return task_interface, device_tensor


def make_tensor_arg(arg: Any) -> Any:
    """Convert an orchestration tensor argument into a simpler ``Tensor``.

    Args:
        arg: One of:
            - ``torch.Tensor``: a CPU-contiguous, pre-fork host tensor named
              through the active simpler Worker.
            - :class:`~pypto.runtime.DeviceTensor`: a worker-resident buffer;
              resolved to its runtime-owned ``Buffer`` and self-describing
              ``Tensor`` view.
            - simpler ``Tensor``: returned as-is (already device-side).

    Returns:
        A simpler ``Tensor`` ready to add to ``TaskArgs``.
    """
    task_interface, device_tensor = _modules()

    if isinstance(arg, task_interface.Tensor):
        return arg
    if isinstance(arg, device_tensor.DeviceTensor):
        resolver = _DEVICE_RESOLVER.get()
        if resolver is None:
            raise RuntimeError("DeviceTensor arguments require a prepared DistributedWorker")
        return resolver(arg)
    worker = _ACTIVE_WORKER.get()
    if worker is None:
        raise RuntimeError("host tensor argument conversion requires an active distributed Worker")
    return task_interface.make_tensor_arg(worker, arg)
