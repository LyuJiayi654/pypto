# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify the pypto-owned ``make_tensor_arg`` used by generated distributed
orchestration code.

It must:
- resolve a worker-resident :class:`DeviceTensor` through the active
  ``DistributedWorker``;
- pass an already-built ``Tensor`` through unchanged;
- delegate a host ``torch.Tensor`` with the active simpler Worker.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor

# ``task_interface`` eagerly imports the optional ``simpler`` runtime package;
# skip the module when simpler is unavailable (same pattern as
# test_execute_compiled_device_tensor.py).
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True

pytestmark = pytest.mark.skipif(not _has_simpler, reason="make_tensor_arg requires the simpler package")


def test_device_tensor_uses_bound_resolver():
    dt = DeviceTensor(0xABCD, (8, 16), torch.float16)
    sentinel = MagicMock(name="Tensor(device)")
    resolver = MagicMock(return_value=sentinel)
    worker = MagicMock(name="Worker(level=3)")

    from pypto.runtime.tensor_arg import bind_tensor_arg_context, make_tensor_arg  # noqa: PLC0415

    with bind_tensor_arg_context(worker, resolver):
        result = make_tensor_arg(dt)

    resolver.assert_called_once_with(dt)
    assert result is sentinel


def test_continuous_tensor_passes_through():
    from pypto.runtime import task_interface  # noqa: PLC0415
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    class FakeTensor:
        pass

    tensor = FakeTensor()
    with patch.object(task_interface, "Tensor", FakeTensor):
        assert make_tensor_arg(tensor) is tensor


def test_host_tensor_delegates_to_simpler():
    host = torch.zeros(4, 4, dtype=torch.float32)
    sentinel = MagicMock(name="Tensor(host)")
    worker = MagicMock(name="Worker(level=3)")

    with patch("pypto.runtime.task_interface.make_tensor_arg", return_value=sentinel) as impl:
        from pypto.runtime.tensor_arg import bind_tensor_arg_context, make_tensor_arg  # noqa: PLC0415

        with bind_tensor_arg_context(worker):
            result = make_tensor_arg(host)

    impl.assert_called_once_with(worker, host)
    assert result is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
