# cuda_ops.py

# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any
import math

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from . import operators  # Import operators for function matching

FakeCUDAKernel = Any

# This code will CUDA compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator that JIT-compiles a function to run on CUDA device."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Wrapper around numba.cuda.jit that handles type annotations."""
    return _jit(**kwargs)(fn)  # type: ignore


# Compile the tensor_data functions for use in CUDA kernels
to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Dispatch to the appropriate CUDA map function based on `fn`."""
        if fn == operators.neg:
            return CudaOps.neg_map()
        elif fn == operators.square:
            return CudaOps.square_map()
        elif fn == operators.sigmoid:
            return CudaOps.sigmoid_map()
        elif fn == operators.relu:
            return CudaOps.relu_map()
        elif fn == operators.inv:
            return CudaOps.inv_map()
        elif fn == operators.exp:
            return CudaOps.exp_map()
        elif fn == operators.log:
            return CudaOps.log_map()
        elif hasattr(fn, 'constant'):
            # Handle functions with constants (e.g., addConstant, mulConstant)
            if fn.__name__ == 'addConstant':
                return CudaOps.add_constant_map(fn.constant)
            elif fn.__name__ == 'mulConstant':
                return CudaOps.mul_constant_map(fn.constant)
            elif fn.__name__ == 'subConstant':
                return CudaOps.sub_constant_map(fn.constant)
        elif fn.__name__ == 'cube':
            return CudaOps.cube_map()
        else:
            raise NotImplementedError(f"Function {fn} is not implemented in CUDA.")

    # Map methods

    @staticmethod
    def neg_map() -> MapProto:
        """CUDA implementation of negation."""
        @cuda.jit()
        def cuda_neg_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = -in_storage[in_pos]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_neg_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def square_map() -> MapProto:
        """CUDA implementation of square."""
        @cuda.jit()
        def cuda_square_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = val * val

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_square_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def cube_map() -> MapProto:
        """CUDA implementation of cube."""
        @cuda.jit()
        def cuda_cube_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = val * val * val

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_cube_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def inv_map() -> MapProto:
        """CUDA implementation of inversion."""
        @cuda.jit()
        def cuda_inv_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = 1.0 / val

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_inv_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def relu_map() -> MapProto:
        """CUDA implementation of ReLU."""
        @cuda.jit()
        def cuda_relu_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = val if val > 0 else 0.0

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_relu_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def sigmoid_map() -> MapProto:
        """CUDA implementation of sigmoid."""
        @cuda.jit()
        def cuda_sigmoid_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = 1.0 / (1.0 + math.exp(-val))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_sigmoid_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def exp_map() -> MapProto:
        """CUDA implementation of exponentiation."""
        @cuda.jit()
        def cuda_exp_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = math.exp(val)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_exp_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def log_map() -> MapProto:
        """CUDA implementation of natural logarithm."""
        @cuda.jit()
        def cuda_log_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                val = in_storage[in_pos]
                out[out_pos] = math.log(val)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_log_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def add_constant_map(constant: float) -> MapProto:
        """CUDA implementation of adding a constant."""
        @cuda.jit()
        def cuda_add_constant_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = in_storage[in_pos] + constant

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_add_constant_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def sub_constant_map(constant: float) -> MapProto:
        """CUDA implementation of subtracting a constant."""
        @cuda.jit()
        def cuda_sub_constant_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = in_storage[in_pos] - constant

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_sub_constant_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    @staticmethod
    def mul_constant_map(constant: float) -> MapProto:
        """CUDA implementation of multiplying by a constant."""
        @cuda.jit()
        def cuda_mul_constant_kernel(
            out, out_shape, out_strides, out_size,
            in_storage, in_shape, in_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = in_storage[in_pos] * constant

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_mul_constant_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple()
            )
            return out

        return ret

    # Zip methods

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Dispatch to the appropriate CUDA zip function based on `fn`."""
        if fn == operators.add:
            return CudaOps.add_zip()
        elif fn == operators.mul:
            return CudaOps.mul_zip()
        elif fn == operators.div:
            return CudaOps.div_zip()
        else:
            raise NotImplementedError(f"Function {fn} is not implemented in CUDA.")

    @staticmethod
    def add_zip() -> Callable[[Tensor, Tensor], Tensor]:
        """CUDA implementation of element-wise addition."""
        @cuda.jit()
        def cuda_add_kernel(
            out, out_shape, out_strides, out_size,
            a_storage, a_shape, a_strides,
            b_storage, b_shape, b_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                a_index = cuda.local.array(MAX_DIMS, numba.int32)
                b_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[out_pos] = a_storage[a_pos] + b_storage[b_pos]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_add_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def mul_zip() -> Callable[[Tensor, Tensor], Tensor]:
        """CUDA implementation of element-wise multiplication."""
        @cuda.jit()
        def cuda_mul_kernel(
            out, out_shape, out_strides, out_size,
            a_storage, a_shape, a_strides,
            b_storage, b_shape, b_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                a_index = cuda.local.array(MAX_DIMS, numba.int32)
                b_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[out_pos] = a_storage[a_pos] * b_storage[b_pos]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_mul_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def div_zip() -> Callable[[Tensor, Tensor], Tensor]:
        """CUDA implementation of element-wise division."""
        @cuda.jit()
        def cuda_div_kernel(
            out, out_shape, out_strides, out_size,
            a_storage, a_shape, a_strides,
            b_storage, b_shape, b_strides
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                a_index = cuda.local.array(MAX_DIMS, numba.int32)
                b_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[out_pos] = a_storage[a_pos] / b_storage[b_pos]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_div_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    # Reduce methods

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Dispatch to the appropriate CUDA reduce function based on `fn`."""
        if fn == operators.add:
            return CudaOps.add_reduce(start)
        else:
            raise NotImplementedError(f"Function {fn} is not implemented in CUDA.")

    @staticmethod
    def add_reduce(start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        """CUDA implementation of sum reduction."""
        @cuda.jit()
        def cuda_add_reduce_kernel(
            out, out_shape, out_strides, out_size,
            a_storage, a_shape, a_strides,
            reduce_dim, start
        ):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i < out_size:
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                a_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(i, out_shape, out_index)

                acc = start
                for j in range(a_shape[reduce_dim]):
                    for k in range(len(out_shape)):
                        a_index[k] = out_index[k]
                    a_index[reduce_dim] = j
                    a_pos = index_to_position(a_index, a_strides)
                    acc += a_storage[a_pos]
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = acc

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
            cuda_add_reduce_kernel[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), dim, start
            )
            return out

        return ret

    # Matrix multiplication

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Compute matrix multiplication of two tensors on CUDA."""
        if len(a.shape) == 2:
            a = a.view(1, *a.shape)
        if len(b.shape) == 2:
            b = b.view(1, *b.shape)

        batch_size = max(a.shape[0], b.shape[0])
        assert a.shape[-1] == b.shape[-2], "Shapes do not align for matrix multiplication."

        out_shape = (batch_size, a.shape[-2], b.shape[-1])
        out = a.zeros(out_shape)

        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        blockspergrid = (
            (out.shape[1] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (out.shape[2] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            batch_size,
        )

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        if a.shape[0] == 1 and b.shape[0] == 1:
            out = out.view(out.shape[1], out.shape[2])

        return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function."""
    batch = cuda.blockIdx.z

    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    M = out_shape[-2]
    N = out_shape[-1]
    K = a_shape[-1]

    if row >= M or col >= N:
        return

    acc = 0.0
    for k in range(K):
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        a_index[0] = batch
        a_index[1] = row
        a_index[2] = k

        b_index[0] = batch
        b_index[1] = k
        b_index[2] = col

        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)

        acc += a_storage[a_pos] * b_storage[b_pos]

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    out_index[0] = batch
    out_index[1] = row
    out_index[2] = col
    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = acc


tensor_matrix_multiply = cuda.jit()(_tensor_matrix_multiply)
