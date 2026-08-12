# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the chip-POD to host-dispatch argument boundary."""

import pytest
import torch

try:
    from simpler.buffer import (  # pyright: ignore[reportMissingImports]
        mint_owner_instance_id,
        wrap_device_malloc,
    )
    from simpler.task_interface import (  # pyright: ignore[reportMissingImports]
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
        TaskArgs,
    )
    from simpler.worker import Worker  # pyright: ignore[reportMissingImports]
    from simpler_setup.torch_interop import make_chip_tensor_arg  # pyright: ignore[reportMissingImports]
except ImportError:
    pytest.skip("worker argument conversion requires the simpler package", allow_module_level=True)

from pypto.runtime import to_worker_task_args


def test_host_chip_args_are_rebuilt_as_self_describing_task_args():
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    chip_args = ChipStorageTaskArgs()
    chip_args.add_tensor(make_chip_tensor_arg(host))
    chip_args.add_scalar(17)

    worker = Worker(level=2)
    task_args = to_worker_task_args(worker, chip_args)

    assert isinstance(task_args, TaskArgs)
    assert task_args.tensor_count() == 1
    assert task_args.scalar_count() == 1
    assert task_args.scalar(0) == 17
    tensor = task_args.tensor(0)
    assert tensor.shapes == (3, 4)
    assert tensor.strides == (4, 1)
    assert tensor.buffer.nbytes == host.numel() * host.element_size()
    assert int.from_bytes(tensor.buffer.body[:8], "little") == host.data_ptr()


def test_device_chip_arg_reuses_the_owning_worker_buffer_identity():
    ptr = 0x1234_0000
    handle = wrap_device_malloc(
        ptr,
        32,
        mint_owner_instance_id(),
        buffer_id=9,
        owner_worker_id=0,
    )
    chip_args = ChipStorageTaskArgs()
    chip_args.add_tensor(ChipTensor.make(ptr, (8,), DataType.FLOAT32, child_memory=True))

    task_args = to_worker_task_args(object(), chip_args, {(0, ptr): handle})

    tensor = task_args.tensor(0)
    assert tensor.buffer.identity == handle.identity
    assert tensor.buffer.body[:8] == ptr.to_bytes(8, "little")
    assert tensor.shapes == (8,)


def test_device_chip_arg_requires_a_live_active_worker_allocation():
    ptr = 0x5678_0000
    chip_args = ChipStorageTaskArgs()
    chip_args.add_tensor(ChipTensor.make(ptr, (4,), DataType.FLOAT32, child_memory=True))

    with pytest.raises(ValueError, match="not a live allocation owned by the active ChipWorker"):
        to_worker_task_args(object(), chip_args)
