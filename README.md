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


# 3.5
simple CPU:

Epoch  0  loss  4.996139797477641 correct 23
Epoch  10  loss  2.8015168106786215 correct 48
Epoch  20  loss  1.9786041528810303 correct 48
Epoch  30  loss  1.6330693467697195 correct 48
Epoch  40  loss  0.11273590810362955 correct 48
Epoch  50  loss  0.10854850289145695 correct 49
Epoch  60  loss  0.8673350917767426 correct 49
Epoch  70  loss  0.03652166470195926 correct 48
Epoch  80  loss  1.2714580513494371 correct 48
Epoch  90  loss  0.19861780175424254 correct 49
Epoch  100  loss  0.28872997287018215 correct 47
Epoch  110  loss  0.6785626055634877 correct 49
Epoch  120  loss  2.1780424285482702 correct 47
Epoch  130  loss  0.2039828778129387 correct 48
Epoch  140  loss  0.027412989191942794 correct 48
Epoch  150  loss  0.6128538280302749 correct 49
Epoch  160  loss  1.0397737938726164 correct 49
Epoch  170  loss  1.6517583808951106 correct 49
Epoch  180  loss  0.015341124372633087 correct 49
Epoch  190  loss  0.02152985966038121 correct 48
Epoch  200  loss  0.6797649305589851 correct 49
Epoch  210  loss  1.177344230126056 correct 49
Epoch  220  loss  0.8466311873914301 correct 48
Epoch  230  loss  2.113962738610868 correct 49
Epoch  240  loss  1.024018816563798 correct 49
Epoch  250  loss  0.029720643241790395 correct 48
Epoch  260  loss  0.9016252862437734 correct 49
Epoch  270  loss  2.0895434800832566 correct 47
Epoch  280  loss  2.2748740258727436 correct 47
Epoch  290  loss  0.13665161305299092 correct 47
Epoch  300  loss  1.7412202328056883 correct 48
Epoch  310  loss  0.3863762488497628 correct 48
Epoch  320  loss  0.6943958998247061 correct 49
Epoch  330  loss  0.0019477000327005787 correct 47
Epoch  340  loss  0.004451546067046999 correct 49
Epoch  350  loss  1.3736670829414033 correct 49
Epoch  360  loss  0.6974702639739182 correct 49
Epoch  370  loss  1.5393218505407529 correct 47
Epoch  380  loss  0.11019570276493497 correct 47
Epoch  390  loss  0.27965595793003795 correct 48
Epoch  400  loss  1.4909901502436944 correct 49
Epoch  410  loss  0.06945750162186046 correct 48
Epoch  420  loss  0.004376092265043443 correct 49
Epoch  430  loss  2.093426673556762 correct 50
Epoch  440  loss  0.3056674391558939 correct 47
Epoch  450  loss  0.003014746756102491 correct 49
Epoch  460  loss  0.8197385381637577 correct 49
Epoch  470  loss  0.0014821186170803865 correct 49
Epoch  480  loss  0.059965511776465 correct 49
Epoch  490  loss  1.7499528823031594 correct 49

Simple GPU:

Epoch  0  loss  3.433547404505915 correct 45
Epoch  10  loss  1.6555451790242222 correct 49
Epoch  20  loss  0.9332955687056259 correct 49
Epoch  30  loss  0.4478020968612947 correct 48
Epoch  40  loss  0.5736978026703554 correct 50
Epoch  50  loss  0.5889392554259449 correct 50
Epoch  60  loss  0.23076967132890777 correct 50
Epoch  70  loss  0.028523361218272522 correct 50
Epoch  80  loss  0.10850970922896883 correct 50
Epoch  90  loss  0.40863398611171414 correct 49
Epoch  100  loss  0.15970103325756813 correct 50
Epoch  110  loss  0.04164585420628033 correct 50
Epoch  120  loss  0.38319352996231165 correct 50
Epoch  130  loss  0.7508758696453952 correct 50
Epoch  140  loss  0.017889984243323994 correct 49
Epoch  150  loss  0.31180142194724836 correct 49
Epoch  160  loss  0.012541975884313623 correct 50
Epoch  170  loss  0.03589984706716991 correct 50
Epoch  180  loss  0.46903406285628546 correct 50
Epoch  190  loss  0.021864406065704854 correct 50
Epoch  200  loss  0.7769745461518809 correct 50
Epoch  210  loss  0.5088475352347304 correct 50
Epoch  220  loss  0.02720390244205873 correct 50
Epoch  230  loss  0.013667775644583689 correct 50
Epoch  240  loss  0.0046443249697793775 correct 50
Epoch  250  loss  0.002560949497775029 correct 50
Epoch  260  loss  0.1518372013482956 correct 50
Epoch  270  loss  0.4480815381903606 correct 50
Epoch  280  loss  0.3811470690351355 correct 50
Epoch  290  loss  0.013256917349291584 correct 50
Epoch  300  loss  0.502421134501242 correct 50
Epoch  310  loss  0.802991066768894 correct 50
Epoch  320  loss  0.4217819017723122 correct 50
Epoch  330  loss  0.8957867667221757 correct 49
Epoch  340  loss  0.35676299827400215 correct 50
Epoch  350  loss  0.007872812531131148 correct 50
Epoch  360  loss  5.3778761153499515e-05 correct 50
Epoch  370  loss  0.5137921402665142 correct 50
Epoch  380  loss  0.23657889775991892 correct 50
Epoch  390  loss  0.37125321721923527 correct 50
Epoch  400  loss  0.007091789236217666 correct 50
Epoch  410  loss  0.00034914633222864124 correct 50
Epoch  420  loss  0.4981216202372262 correct 50
Epoch  430  loss  0.3556113230034635 correct 50
Epoch  440  loss  0.011074430631313326 correct 50
Epoch  450  loss  0.0008531153745036555 correct 50
Epoch  460  loss  0.3314444264660943 correct 50
Epoch  470  loss  0.003119461090917717 correct 50
Epoch  480  loss  0.2452325693927098 correct 50
Epoch  490  loss  0.00048738493890668026 correct 50


Split CPU:

Epoch  0  loss  5.475210153939104 correct 34
Epoch  10  loss  6.402111462612603 correct 45
Epoch  20  loss  4.92482340738525 correct 44
Epoch  30  loss  5.996763150597838 correct 43
Epoch  40  loss  3.378424771387845 correct 45
Epoch  50  loss  3.392762612480446 correct 44
Epoch  60  loss  1.6235356159189815 correct 46
Epoch  70  loss  2.4164560023406874 correct 45
Epoch  80  loss  3.7649646277249604 correct 46
Epoch  90  loss  2.131265621853718 correct 46
Epoch  100  loss  1.7645758979135249 correct 47
Epoch  110  loss  2.2059635065989576 correct 48
Epoch  120  loss  2.68424480180808 correct 50
Epoch  130  loss  3.188400182766273 correct 46
Epoch  140  loss  0.8496055688160243 correct 49
Epoch  150  loss  1.138592501380384 correct 49
Epoch  160  loss  0.7793770689440064 correct 49
Epoch  170  loss  1.4521573148207645 correct 50
Epoch  180  loss  2.927356375746046 correct 45
Epoch  190  loss  1.2675474113698533 correct 49
Epoch  200  loss  1.075454051852885 correct 50
Epoch  210  loss  1.6424019014408748 correct 49
Epoch  220  loss  0.8474352232685554 correct 50
Epoch  230  loss  1.6898209876404144 correct 48
Epoch  240  loss  0.9057270815292771 correct 50
Epoch  250  loss  0.9681122063065826 correct 50
Epoch  260  loss  0.7149116171352206 correct 50
Epoch  270  loss  1.1194512948049093 correct 49
Epoch  280  loss  0.31693891255896156 correct 49
Epoch  290  loss  1.5868819026326217 correct 49
Epoch  300  loss  0.43425406694263674 correct 50
Epoch  310  loss  0.7183313125533802 correct 50
Epoch  320  loss  0.19665160080647986 correct 49
Epoch  330  loss  1.1006990220325559 correct 50
Epoch  340  loss  0.5264848857903243 correct 50
Epoch  350  loss  0.46128598489932876 correct 50
Epoch  360  loss  0.08369755316356392 correct 50
Epoch  370  loss  0.22002342825585702 correct 50
Epoch  380  loss  0.33832728050402805 correct 50
Epoch  390  loss  0.47881139250859117 correct 50
Epoch  400  loss  1.1129804669308552 correct 50
Epoch  410  loss  0.9097524851842094 correct 50
Epoch  420  loss  0.5240465666996224 correct 50
Epoch  430  loss  0.6410437231356898 correct 50
Epoch  440  loss  0.42216958810702515 correct 50
Epoch  450  loss  0.09003319122498256 correct 50
Epoch  460  loss  0.7786313038628435 correct 50
Epoch  470  loss  0.44934826477649886 correct 50
Epoch  480  loss  0.3240498335129722 correct 50
Epoch  490  loss  0.5065479079486894 correct 50

Split GPU:
Epoch  0  loss  6.765574359349766 correct 33
Epoch  10  loss  6.775217686024695 correct 41
Epoch  20  loss  6.133004339848437 correct 43
Epoch  30  loss  4.60989896265142 correct 47
Epoch  40  loss  4.609592619493904 correct 43
Epoch  50  loss  3.8117640176509235 correct 48
Epoch  60  loss  3.102623743810108 correct 48
Epoch  70  loss  3.127589338785118 correct 46
Epoch  80  loss  2.2174543240788536 correct 48
Epoch  90  loss  0.960639294350454 correct 48
Epoch  100  loss  2.106258520747005 correct 47
Epoch  110  loss  1.3297898517553504 correct 49
Epoch  120  loss  1.9630939377549645 correct 50
Epoch  130  loss  0.7252385085906067 correct 48
Epoch  140  loss  2.1318567783164104 correct 50
Epoch  150  loss  1.1242461754724586 correct 48
Epoch  160  loss  0.5750081403241861 correct 50
Epoch  170  loss  0.3495487130013659 correct 49
Epoch  180  loss  1.6021126174324312 correct 50
Epoch  190  loss  1.0185725866738395 correct 49
Epoch  200  loss  1.8295399242611476 correct 49
Epoch  210  loss  0.41842652976183087 correct 50
Epoch  220  loss  0.6022353973849096 correct 49
Epoch  230  loss  0.8562489302235478 correct 49
Epoch  240  loss  0.1892804674872115 correct 50
Epoch  250  loss  0.3608263795033617 correct 50
Epoch  260  loss  0.4664622872637828 correct 50
Epoch  270  loss  0.7965160696593334 correct 50
Epoch  280  loss  0.7411072678451511 correct 50
Epoch  290  loss  0.28405419133396875 correct 50
Epoch  300  loss  0.11755667214519565 correct 50
Epoch  310  loss  0.1863872028335693 correct 50
Epoch  320  loss  0.4504250222895915 correct 50
Epoch  330  loss  0.28666766327782467 correct 50
Epoch  340  loss  0.9939107552901991 correct 50
Epoch  350  loss  0.40701357267237154 correct 50
Epoch  360  loss  0.06181211414747045 correct 50
Epoch  370  loss  0.767085053540079 correct 50
Epoch  380  loss  0.6195510800013717 correct 50
Epoch  390  loss  0.2253235650480756 correct 50
Epoch  400  loss  0.06896608761031656 correct 50
Epoch  410  loss  0.10434241483260495 correct 49
Epoch  420  loss  0.41533077894681203 correct 50
Epoch  430  loss  0.7734220087309984 correct 50
Epoch  440  loss  0.07046787168086582 correct 50
Epoch  450  loss  0.15289310300877984 correct 50
Epoch  460  loss  0.24053989936275183 correct 50
Epoch  470  loss  0.028824111720629975 correct 50
Epoch  480  loss  0.012700396528031552 correct 50
Epoch  490  loss  0.011332662496578348 correct 50


XOR CPU:

Epoch  0  loss  6.3515999223518165 correct 24
Epoch  10  loss  4.71260853643058 correct 41
Epoch  20  loss  3.684696485982372 correct 44
Epoch  30  loss  4.784776214424528 correct 45
Epoch  40  loss  2.438376293596356 correct 44
Epoch  50  loss  2.2094887494560513 correct 45
Epoch  60  loss  1.8214624042991463 correct 45
Epoch  70  loss  1.4137425256289187 correct 45
Epoch  80  loss  1.8108069493654586 correct 45
Epoch  90  loss  2.8304341106242545 correct 48
Epoch  100  loss  1.89103077907573 correct 45
Epoch  110  loss  3.7958144278551775 correct 47
Epoch  120  loss  1.4341897211128325 correct 47
Epoch  130  loss  2.140597635358252 correct 49
Epoch  140  loss  1.7044177170079935 correct 48
Epoch  150  loss  2.5171487865338693 correct 50
Epoch  160  loss  0.9105705829631661 correct 50
Epoch  170  loss  1.7944183870116304 correct 47
Epoch  180  loss  0.6964848165906405 correct 50
Epoch  190  loss  1.1583923854291809 correct 49
Epoch  200  loss  0.907449570617392 correct 49
Epoch  210  loss  1.03131497067035 correct 49
Epoch  220  loss  0.7235915206873691 correct 50
Epoch  230  loss  0.3922885559892321 correct 50
Epoch  240  loss  0.6495436013510985 correct 50
Epoch  250  loss  0.6338464428941012 correct 50
Epoch  260  loss  0.7086941005761863 correct 50
Epoch  270  loss  0.3639913718643078 correct 50
Epoch  280  loss  0.8411247497369237 correct 50
Epoch  290  loss  0.7353415750701118 correct 50
Epoch  300  loss  0.5191328339763889 correct 50
Epoch  310  loss  0.32903318156682837 correct 50
Epoch  320  loss  0.42381248505597663 correct 50
Epoch  330  loss  0.3935181618577477 correct 50
Epoch  340  loss  0.7647951677933932 correct 50
Epoch  350  loss  0.3057660774618405 correct 50
Epoch  360  loss  0.4702802803018704 correct 50
Epoch  370  loss  0.399653282624707 correct 50
Epoch  380  loss  0.20082980713564835 correct 50
Epoch  390  loss  0.2941935391423966 correct 50
Epoch  400  loss  0.4013414351329052 correct 50
Epoch  410  loss  0.16925807197160875 correct 50
Epoch  420  loss  0.18241182875054676 correct 50
Epoch  430  loss  0.21590163860569217 correct 50
Epoch  440  loss  0.24502267582481327 correct 50
Epoch  450  loss  0.1428802697276496 correct 50
Epoch  460  loss  0.4192985440447296 correct 50
Epoch  470  loss  0.1666661805618535 correct 50
Epoch  480  loss  0.08846861549126486 correct 50
Epoch  490  loss  0.3456746446755993 correct 50


XOR GPU:

Epoch  0  loss  5.696260255961773 correct 30
Epoch  10  loss  4.721450176345213 correct 46
Epoch  20  loss  3.7847344116124755 correct 46
Epoch  30  loss  2.572576918919909 correct 38
Epoch  40  loss  2.21055017950886 correct 46
Epoch  50  loss  3.4085929425652766 correct 46
Epoch  60  loss  1.3557140126218266 correct 46
Epoch  70  loss  3.9686593956390794 correct 46
Epoch  80  loss  1.0026770398216855 correct 46
Epoch  90  loss  3.4006606391714693 correct 45
Epoch  100  loss  1.723260127119524 correct 46
Epoch  110  loss  2.5018562335233137 correct 48
Epoch  120  loss  1.0973874192195527 correct 45
Epoch  130  loss  1.7329229313777552 correct 46
Epoch  140  loss  1.5788927162005781 correct 47
Epoch  150  loss  0.6300466053321806 correct 47
Epoch  160  loss  0.42534282297779913 correct 47
Epoch  170  loss  0.8767268946994744 correct 48
Epoch  180  loss  0.3653695368147444 correct 48
Epoch  190  loss  1.8849046497823028 correct 48
Epoch  200  loss  1.6037668632930537 correct 48
Epoch  210  loss  0.9435388228010959 correct 49
Epoch  220  loss  1.9856667701447348 correct 46
Epoch  230  loss  1.8750095955603339 correct 46
Epoch  240  loss  0.9857502186763223 correct 48
Epoch  250  loss  2.1270986794839972 correct 49
Epoch  260  loss  0.4113738927981139 correct 49
Epoch  270  loss  0.5564881722332309 correct 49
Epoch  280  loss  0.7544068885752578 correct 49
Epoch  290  loss  0.4580394337830654 correct 49
Epoch  300  loss  0.7669535954543844 correct 49
Epoch  310  loss  1.3308507741715219 correct 49
Epoch  320  loss  1.144012804034248 correct 49
Epoch  330  loss  0.6577630370716648 correct 49
Epoch  340  loss  0.16534613504751017 correct 49
Epoch  350  loss  1.013104451180681 correct 49
Epoch  360  loss  1.3975069609342596 correct 49
Epoch  370  loss  0.3485268005064225 correct 49
Epoch  380  loss  1.6693129003115144 correct 50
Epoch  390  loss  0.1859908027107029 correct 50
Epoch  400  loss  0.22125950822528412 correct 50
Epoch  410  loss  0.4940634223468887 correct 49
Epoch  420  loss  0.888869482900067 correct 49
Epoch  430  loss  0.33449077219426004 correct 50
Epoch  440  loss  1.008577642758182 correct 49
Epoch  450  loss  0.3548421170237432 correct 49
Epoch  460  loss  0.6273572956426121 correct 49
Epoch  470  loss  0.13891109822566586 correct 49
Epoch  480  loss  1.045941819883647 correct 50
Epoch  490  loss  0.36225428120599656 correct 50

# Bigger Model

Split CPU 500:

Epoch  0  loss  139.7409738357781 correct 31
Epoch  10  loss  5.456989422151462 correct 31
Epoch  20  loss  3.529833599984782 correct 49
Epoch  30  loss  0.9435482816331086 correct 48
Epoch  40  loss  2.588507806588864 correct 47
Epoch  50  loss  1.542105501701858 correct 50
Epoch  60  loss  0.5569494331923978 correct 50
Epoch  70  loss  2.8991883405498546 correct 49
Epoch  80  loss  0.3940031437542586 correct 49
Epoch  90  loss  0.6744291202919734 correct 50
Epoch  100  loss  0.8091754736869893 correct 49
Epoch  110  loss  0.1446713189292921 correct 49
Epoch  120  loss  0.8804045656947883 correct 49
Epoch  130  loss  0.3891960093501275 correct 50
Epoch  140  loss  0.4507112299212169 correct 49
Epoch  150  loss  2.271194922558069 correct 49
Epoch  160  loss  0.3945861362064735 correct 49
Epoch  170  loss  1.0945362985086464 correct 49
Epoch  180  loss  0.7945734924555017 correct 49
Epoch  190  loss  0.5044598051905764 correct 50
Epoch  200  loss  0.5582316348697158 correct 49
Epoch  210  loss  0.20703250837652978 correct 48
Epoch  220  loss  0.07670079310916125 correct 50
Epoch  230  loss  0.13265282942353473 correct 49
Epoch  240  loss  0.12292026069816293 correct 49
Epoch  250  loss  2.3623666027091135 correct 48
Epoch  260  loss  0.41489291496607444 correct 49
Epoch  270  loss  0.1847509278141991 correct 50
Epoch  280  loss  0.13948362684708893 correct 49
Epoch  290  loss  0.20775956374628002 correct 49
Epoch  300  loss  0.8783010544350565 correct 50
Epoch  310  loss  0.9350112090225098 correct 50
Epoch  320  loss  0.27369855699892526 correct 50
Epoch  330  loss  0.007050842633418799 correct 49
Epoch  340  loss  0.21762298103453917 correct 49
Epoch  350  loss  0.1744345805278922 correct 49
Epoch  360  loss  0.1892651590360391 correct 49
Epoch  370  loss  0.2999390642459274 correct 50
Epoch  380  loss  0.15743392259584882 correct 49
Epoch  390  loss  0.06951841307842499 correct 49
Epoch  400  loss  0.06509495841334201 correct 49
Epoch  410  loss  0.1212439083371677 correct 49
Epoch  420  loss  0.39392759974837555 correct 49
Epoch  430  loss  2.0149815654415817 correct 49
Epoch  440  loss  1.0370336395806243 correct 49
Epoch  450  loss  0.9702703687650613 correct 49
Epoch  460  loss  2.2582395381346987 correct 48
Epoch  470  loss  0.04787114102704488 correct 49
Epoch  480  loss  0.15938468311922796 correct 49
Epoch  490  loss  0.15816266780808763 correct 49

Split GPU 500:

Epoch  0  loss  77.56459731610009 correct 26
Epoch  10  loss  6.213767518769 correct 38
Epoch  20  loss  0.21985241074601816 correct 49
Epoch  30  loss  0.8250498395994985 correct 49
Epoch  40  loss  0.8325544831474472 correct 50
Epoch  50  loss  0.6394798902957695 correct 50
Epoch  60  loss  0.5324457553810094 correct 50
Epoch  70  loss  0.1469781665040507 correct 50
Epoch  80  loss  0.38395210068986557 correct 50
Epoch  90  loss  0.22614570686035204 correct 50
Epoch  100  loss  0.3122512321014956 correct 50
Epoch  110  loss  0.2058986065014633 correct 50
Epoch  120  loss  0.18428620139792742 correct 50
Epoch  130  loss  0.16023043546929225 correct 50
Epoch  140  loss  0.13807197138777158 correct 50
Epoch  150  loss  0.3325548127553296 correct 50
Epoch  160  loss  0.11847422847878168 correct 50
Epoch  170  loss  0.09236888227440233 correct 50
Epoch  180  loss  0.17800670050138995 correct 50
Epoch  190  loss  0.11980427493987052 correct 50
Epoch  200  loss  0.02744290417038536 correct 50
Epoch  210  loss  0.12384122391024044 correct 50
Epoch  220  loss  0.13100926983398528 correct 50
Epoch  230  loss  0.12965402342002658 correct 50
Epoch  240  loss  0.07349134282280115 correct 50
Epoch  250  loss  0.08333121727590255 correct 50
Epoch  260  loss  0.07538513837131602 correct 50
Epoch  270  loss  0.04474817429596593 correct 50
Epoch  280  loss  0.057065146002506645 correct 50
Epoch  290  loss  0.04011390177177959 correct 50
Epoch  300  loss  0.10170315468221039 correct 50
Epoch  310  loss  0.12251325132174712 correct 50
Epoch  320  loss  0.11952118555199182 correct 50
Epoch  330  loss  0.07399934954126468 correct 50
Epoch  340  loss  0.07571498313889892 correct 50
Epoch  350  loss  0.03430646310534795 correct 50
Epoch  360  loss  0.06813443323489136 correct 50
Epoch  370  loss  0.027728484071110983 correct 50
Epoch  380  loss  0.05967904469913386 correct 50
Epoch  390  loss  0.08730162618528578 correct 50
Epoch  400  loss  0.021115628650148703 correct 50
Epoch  410  loss  0.07063459666001645 correct 50
Epoch  420  loss  0.05617020444448072 correct 50
Epoch  430  loss  0.0780968149836553 correct 50
Epoch  440  loss  0.028559553032887475 correct 50
Epoch  450  loss  0.037110765089145986 correct 50
Epoch  460  loss  0.014232935777467901 correct 50
Epoch  470  loss  0.02020715649821079 correct 50
Epoch  480  loss  0.005876711556258353 correct 50
Epoch  490  loss  0.02063880564100175 correct 50
