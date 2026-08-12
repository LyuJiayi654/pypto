# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``ChipWorker`` device-memory primitives and ``alloc_tensor``.

Patches ``_SimplerWorker`` so tests run without a device.  Each test asserts
that the call is forwarded to the underlying simpler worker with the expected
runtime-owned ``Buffer`` handles.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so ChipWorker construction does not touch a device."""
    with (
        patch("pypto.runtime.worker._SimplerWorker") as cls,
        # init() builds a prewarm CallConfig; patch the cache so no simpler import happens.
        patch("pypto.runtime.worker._SimplerCallConfig", MagicMock()),
    ):
        instance = MagicMock()
        instance.malloc.side_effect = lambda nbytes: MagicMock(base=0x4000 if nbytes == 1024 else 0x9000)
        cls.return_value = instance
        yield instance


@pytest.fixture
def worker(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    yield w
    if w.initialized:
        w.close()


class TestMallocFree:
    def test_malloc_forwards_with_default_worker_id(self, fake_simpler_worker, worker):
        ptr = worker.malloc(1024)
        assert ptr == 0x4000
        fake_simpler_worker.malloc.assert_called_once_with(1024)

    def test_malloc_forwards_explicit_worker_id(self, fake_simpler_worker, worker):
        with pytest.raises(ValueError, match="worker_id=0"):
            worker.malloc(2048, worker_id=3)
        fake_simpler_worker.malloc.assert_not_called()

    def test_malloc_zero_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(0)

    def test_malloc_negative_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(-1)

    def test_malloc_after_close_raises(self, fake_simpler_worker, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.malloc(1024)
        fake_simpler_worker.malloc.assert_not_called()

    def test_free_forwards(self, fake_simpler_worker, worker):
        worker.malloc(1024)
        handle = worker._device_buffers[(0, 0x4000)]
        worker.free(0x4000)
        fake_simpler_worker.free.assert_called_once_with(handle)

    def test_free_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.free(0x4000)


class TestCopy:
    def test_copy_to_forwards(self, fake_simpler_worker, worker):
        src = torch.arange(64, dtype=torch.uint8)
        ptr = worker.malloc(1024)
        worker.copy_to(ptr, src.data_ptr(), 64)
        handle, payload = fake_simpler_worker.copy_to.call_args.args
        assert handle is worker._device_buffers[(0, ptr)]
        assert len(payload) == 64

    def test_copy_from_forwards(self, fake_simpler_worker, worker):
        dst = torch.empty(64, dtype=torch.uint8)
        ptr = worker.malloc(1024)
        worker.copy_from(dst.data_ptr(), ptr, 64)
        payload, handle = fake_simpler_worker.copy_from.call_args.args
        assert handle is worker._device_buffers[(0, ptr)]
        assert len(payload) == 64

    def test_copy_to_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.copy_to(0x100, 0x200, 64)

    def test_copy_from_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.copy_from(0x100, 0x200, 64)


class TestAllocTensor:
    def test_alloc_no_init(self, fake_simpler_worker, worker):
        t = worker.alloc_tensor((4, 8), torch.float32)
        assert isinstance(t, DeviceTensor)
        assert t.data_ptr == 0x9000
        assert t.shape == (4, 8)
        assert t.dtype is torch.float32
        assert t.nbytes == 4 * 8 * 4
        fake_simpler_worker.malloc.assert_called_once_with(4 * 8 * 4)
        fake_simpler_worker.copy_to.assert_not_called()

    def test_alloc_with_init_uploads(self, fake_simpler_worker, worker):
        host = torch.full((4, 8), 1.5, dtype=torch.float32)
        t = worker.alloc_tensor((4, 8), torch.float32, init=host)
        assert t.data_ptr == 0x9000
        # nbytes is the third positional arg
        fake_simpler_worker.copy_to.assert_called_once()
        call = fake_simpler_worker.copy_to.call_args
        assert call.args[0] is worker._device_buffers[(0, 0x9000)]
        assert len(call.args[1]) == 4 * 8 * 4

    def test_alloc_init_shape_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        bad = torch.zeros((4, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        assert fake_simpler_worker.free.call_args.args[0].base == 0x9000
        fake_simpler_worker.copy_to.assert_not_called()

    def test_alloc_init_dtype_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        bad = torch.zeros((4, 8), dtype=torch.float16)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        assert fake_simpler_worker.free.call_args.args[0].base == 0x9000

    def test_free_tensor_uses_data_ptr(self, fake_simpler_worker, worker):
        # ``free_tensor`` is the dual of ``alloc_tensor``; only tensors the
        # Worker actually allocated are tracked (and therefore freed). Going
        # through alloc_tensor puts the ptr in ``_owned_tensors`` so the
        # subsequent free_tensor forwards through to the underlying ``free``.
        t = worker.alloc_tensor((4, 8), torch.float32)
        fake_simpler_worker.free.reset_mock()
        worker.free_tensor(t)
        assert fake_simpler_worker.free.call_args.args[0].base == 0x9000

    def test_alloc_makes_non_contiguous_init_contiguous(self, fake_simpler_worker, worker):
        # transpose makes it non-contiguous; .contiguous() inside alloc_tensor must fix it.
        host = torch.zeros((8, 4), dtype=torch.float32).t()
        assert tuple(host.shape) == (4, 8)
        worker.alloc_tensor((4, 8), torch.float32, init=host)
        fake_simpler_worker.copy_to.assert_called_once()

    def test_alloc_non_positive_dim_rejected_before_malloc(self, fake_simpler_worker, worker):
        # The shape contract (mirroring DeviceTensor) requires positive int dims.
        # Negative and zero dims must be rejected before any allocation happens —
        # a zero dim would otherwise compute nbytes as 0 (empty shape -> nbytes 1).
        with pytest.raises(ValueError, match="positive"):
            worker.alloc_tensor((-1, 4), torch.float32)
        with pytest.raises(ValueError, match="positive"):
            worker.alloc_tensor((0, 4), torch.float32)
        fake_simpler_worker.malloc.assert_not_called()

    def test_alloc_empty_shape_rejected_before_malloc(self, fake_simpler_worker, worker):
        # An empty shape would make n_elems collapse to 1 and malloc a bogus buffer.
        with pytest.raises(ValueError, match="non-empty"):
            worker.alloc_tensor((), torch.float32)
        fake_simpler_worker.malloc.assert_not_called()

    def test_alloc_copy_to_failure_frees_pointer(self, fake_simpler_worker, worker):
        # Simulate a runtime copy_to failure (hardware error, etc.).  The pointer
        # malloc'd up-front must be freed before the exception propagates so the
        # device buffer is not leaked.
        fake_simpler_worker.copy_to.side_effect = RuntimeError("copy failed")
        host = torch.zeros((4, 8), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="copy failed"):
            worker.alloc_tensor((4, 8), torch.float32, init=host)
        assert fake_simpler_worker.free.call_args.args[0].base == 0x9000

    def test_alloc_tensor_rejects_nonzero_worker_id(self, fake_simpler_worker, worker):
        # ChipWorker owns only worker 0; nonzero worker ids are invalid.
        with pytest.raises(ValueError, match="worker_id=0"):
            worker.alloc_tensor((4, 8), torch.float32, worker_id=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
