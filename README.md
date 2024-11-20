# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# 3.1
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        total_elements = len(out)                                                              | 
                                                                                               | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    | 
            for i in prange(total_elements):---------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                     | 
        else:                                                                                  | 
            for i in prange(total_elements):---------------------------------------------------| #3
                out_index = np.zeros(len(out_shape), np.int32)---------------------------------| #0
                in_index = np.zeros(len(in_shape), np.int32)-----------------------------------| #1
                to_index(i, out_shape, out_index)                                              | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                      | 
                out_pos = index_to_position(out_index, out_strides)                            | 
                in_pos = index_to_position(in_index, in_strides)                               | 
                out[out_pos] = fn(in_storage[in_pos])                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (178) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (179) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (212)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (212) 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                  | 
        out: Storage,                                                                                                                                                          | 
        out_shape: Shape,                                                                                                                                                      | 
        out_strides: Strides,                                                                                                                                                  | 
        a_storage: Storage,                                                                                                                                                    | 
        a_shape: Shape,                                                                                                                                                        | 
        a_strides: Strides,                                                                                                                                                    | 
        b_storage: Storage,                                                                                                                                                    | 
        b_shape: Shape,                                                                                                                                                        | 
        b_strides: Strides,                                                                                                                                                    | 
    ) -> None:                                                                                                                                                                 | 
        total_elements = len(out)                                                                                                                                              | 
                                                                                                                                                                               | 
        if np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides) and np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape):    | 
            for i in prange(total_elements):-----------------------------------------------------------------------------------------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                        | 
        else:                                                                                                                                                                  | 
            for i in prange(total_elements):-----------------------------------------------------------------------------------------------------------------------------------| #8
                out_index = np.zeros(len(out_shape), np.int32)-----------------------------------------------------------------------------------------------------------------| #4
                a_index = np.zeros(len(a_shape), np.int32)---------------------------------------------------------------------------------------------------------------------| #5
                b_index = np.zeros(len(b_shape), np.int32)---------------------------------------------------------------------------------------------------------------------| #6
                to_index(i, out_shape, out_index)                                                                                                                              | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                        | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                        | 
                out_pos = index_to_position(out_index, out_strides)                                                                                                            | 
                a_pos = index_to_position(a_index, a_strides)                                                                                                                  | 
                b_pos = index_to_position(b_index, b_strides)                                                                                                                  | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (230) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (231) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (232) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (264)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (264) 
------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                    | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        a_storage: Storage,                                                         | 
        a_shape: Shape,                                                             | 
        a_strides: Strides,                                                         | 
        reduce_dim: int,                                                            | 
    ) -> None:                                                                      | 
        total_elements = len(out)                                                   | 
        reduce_extent = a_shape[reduce_dim]                                         | 
        reduce_stride = a_strides[reduce_dim]                                       | 
                                                                                    | 
        for idx in prange(total_elements):------------------------------------------| #9
                                                                                    | 
            out_indices = np.empty(len(out_shape), dtype=np.int32)                  | 
            to_index(idx, out_shape, out_indices)                                   | 
                                                                                    | 
            out_position = index_to_position(out_indices, out_strides)              | 
                                                                                    | 
            accumulated_result = out[out_position]                                  | 
                                                                                    | 
            base_pos = index_to_position(out_indices, a_strides)                    | 
                                                                                    | 
            for count in range(reduce_extent):                                      | 
                accumulated_result = fn(accumulated_result, a_storage[base_pos])    | 
                base_pos += reduce_stride                                           | 
                                                                                    | 
            out[out_position] = accumulated_result                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (279) is 
hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_indices = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None


# 3.2
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (296)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/austinl/Desktop/workspace-a2/mod3-awwstin/minitorch/fast_ops.py (296) 
-----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                             | 
    out: Storage,                                                                        | 
    out_shape: Shape,                                                                    | 
    out_strides: Strides,                                                                | 
    a_storage: Storage,                                                                  | 
    a_shape: Shape,                                                                      | 
    a_strides: Strides,                                                                  | 
    b_storage: Storage,                                                                  | 
    b_shape: Shape,                                                                      | 
    b_strides: Strides,                                                                  | 
) -> None:                                                                               | 
    """NUMBA tensor matrix multiply function.                                            | 
                                                                                         | 
    Should work for any tensor shapes that broadcast as long as                          | 
                                                                                         | 
    ```                                                                                  | 
    assert a_shape[-1] == b_shape[-2]                                                    | 
    ```                                                                                  | 
                                                                                         | 
    Optimizations:                                                                       | 
                                                                                         | 
    * Outer loop in parallel                                                             | 
    * No index buffers or function calls                                                 | 
    * Inner loop should have no global writes, 1 multiply.                               | 
                                                                                         | 
                                                                                         | 
    Args:                                                                                | 
    ----                                                                                 | 
        out (Storage): storage for `out` tensor                                          | 
        out_shape (Shape): shape for `out` tensor                                        | 
        out_strides (Strides): strides for `out` tensor                                  | 
        a_storage (Storage): storage for `a` tensor                                      | 
        a_shape (Shape): shape for `a` tensor                                            | 
        a_strides (Strides): strides for `a` tensor                                      | 
        b_storage (Storage): storage for `b` tensor                                      | 
        b_shape (Shape): shape for `b` tensor                                            | 
        b_strides (Strides): strides for `b` tensor                                      | 
                                                                                         | 
    Returns:                                                                             | 
    -------                                                                              | 
        None : Fills in `out`                                                            | 
                                                                                         | 
    """                                                                                  | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               | 
                                                                                         | 
    # Extract sizes and strides for the dimensions                                       | 
    batch_size = out_shape[0]                                                            | 
    rows = out_shape[1]                                                                  | 
    cols = out_shape[2]                                                                  | 
    inner_dim = a_shape[-1]                                                              | 
                                                                                         | 
    # Batch stride to step between matrices for A and B                                  | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               | 
                                                                                         | 
    # Iterate over all elements in the batch and the output matrix                       | 
    for batch in prange(batch_size):-----------------------------------------------------| #10
        out_batch_offset = batch * out_strides[0]                                        | 
        a_batch_offset = batch * a_batch_stride                                          | 
        b_batch_offset = batch * b_batch_stride                                          | 
                                                                                         | 
        for i in range(rows):                                                            | 
            for j in range(cols):                                                        | 
                # Initialize accumulator for the dot product                             | 
                acc = 0.0                                                                | 
                                                                                         | 
                # Calculate starting positions for row in A and column in B              | 
                a_pos = a_batch_offset + i * a_strides[1]                                | 
                b_pos = b_batch_offset + j * b_strides[2]                                | 
                                                                                         | 
                # Inner loop over the shared dimension                                   | 
                for k in range(inner_dim):                                               | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                           | 
                                                                                         | 
                    # Move to the next element in the row of A and column of B           | 
                    a_pos += a_strides[2]                                                | 
                    b_pos += b_strides[1]                                                | 
                                                                                         | 
                # Store the accumulated result in the output                             | 
                out[out_batch_offset + i * out_strides[1] + j * out_strides[2]] = acc    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


# 3.4 Timing Graph
Timing summary
Size: 64
    fast: 0.00363
    gpu: 0.00663
Size: 128
    fast: 0.01672
    gpu: 0.01565
Size: 256
    fast: 0.10108
    gpu: 0.05319
Size: 512
    fast: 1.01162
    gpu: 0.26372
Size: 1024
    fast: 8.31520
    gpu: 1.02751
