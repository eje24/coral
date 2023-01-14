# coral

A simple PyTorch-inspired deep learning framework and tensor library in C.

OVERVIEW OF COMPONENTS:
- variable_t: user-facing object, holds both data (tensor_t*) and grad metadata (grad_meta_t*) allowing for automatic differentiation
- tensor_t: container for raw data and metadata describing size, dimensions, etc
- shape_t: stores metadata describing a chunk of data (num_dims, size, dims, strides)
- grad_meta_t: stores grad-related metadata for a node (variable_t) in the computation graph. explicitly, stores the number of arguments, and an array of diff_arg_t's, one for each argument
- diff_arg_t: an argument with respect to which a given function is differentiable, holds a pointer (variable_t*) to the variable

TODO:
- misc
    - ✅ change name (DONE)
- tensor
    - ✅ Differentiate between backward_grad function and backward function (backward grad takes into account gradient of result to compute gradient of arguments) 
        - for now, change `.._grad` functions to `..._grad_backwards` functions
    - ✅ [#0] Extend broadcasting to more lenient numpy broadcasting scheme where dimensions can be 1 by fixing broadcast logic in tensor.c
    - ✅ [#1] add in ability to construct different views of the same tensor (just have another tensor pointing to the same data, but with different num_rows, num_columns) as well as reductions (sum along dimensions)
    - ✅ Switch naming convention so as to remove function names starting with `_` (see naming convention below)
        - see https://softwareengineering.stackexchange.com/a/115564
    - 🏗️ add new scalar_grad_op function, for functions which use scalars (ie, tensor_divide_by_scalar, etc)
    - 🏗️ add ability to perform operations on various dimensions (mean along dimension zero, axes in numpy)
    - 🏗️ find some graceful way of dealing with unused grad parameters 
        - right now, n-ary functions are assumed to have n-ary gradients, but in many cases the gradient function for a particular variable only involves some subset of the other variables. for example: (d/dx)(x+y) doesn't involve either of x or y. 
    - 🏗️ beautify display functions
    - 🏗️ [#2] add in loss functions (including reductions)
    - 🏗️ [#3] add in matrix multiplications
    - 🏗️ [#4] assert that dimenions are correct/compatible when doing operations
    - 🏗️ Sphinx documentatio (would be cool)
    - 🏗️ Add "fastpath" for broadcasting when two shapes (or shape-suffixes) are the same
    - 🏗️ functions which do not modify should have const arguments (_tensor_add, _tensor_subtract, etc)
    - 🏗️ update _tensor_broadcast_scalar_fn to two versions (binary and unary) (current implementation is binary)
    - 🏗️ standardize naming (child vs parent?? left/right variable/entry/arg??)
        - move toward arg/result naming scheme
    - 🏗️ start test suite
        - add test built target
        - add one main driver which calls test_variable, test_tensor, test_shape, test_grad, etc
        - should at least contain:
            - tests for broadcasting
            - tests for each forwards operation (computes correct result)
            - tests autograd
                - correct graph
                - correctly computes gradients
    - 🏗️ add struct constant_t, and make variable_t an extension
    - 🏗️ byte-align tensor data
    - 🏗️ enable link-time optimization (quick)
    - extend tensor index/entry value lambda broadcasts to variable
    - 🏗️ reference count and "garbage collect" old tensors
    - ✅ shape_t update (for keeping track of tensor dims)
    - 🏗️ migrate to `_tensor_in_place_...` naming convention for in place tensor operatiosn (and variable operations with `_variable_in_place_...`)
    - ℹ️: for now, grad_ops return tensors, not variables, as we do not care about higher order derivatives (i.e. treating gradients as variables in their own right)
    - ✅ make everything heap-allocated
    - ✅ variadic update (tensors should be able to be initialized up to 3 dimensions) - DONE
    - ✅ allow for broadcasting of different but compatible sizes (similar to numpy) - DONE
    - ✅ be consistent about using tensor vs tensor* for static inlined tensor functions (which is better? -> pointers)
    - ✅ add in ability to populate tensor with function pointer
        - ✅ index function
        - ✅ entry value function
- nn
- utils

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
- avoid variadic functions when possible
- one thing which is hard to replicate in C is "self": if a struct has a function, it is not possible to write the function so as to include a referenc "self" to the struct itself
- function/operator overloading would be useful (have to include two separate versions of each function, one for unary, and one for binary)
- union types aren't great
- how should we multiply tensors by a constant? should we construct a new tensor with all entries equal to that constant and then return the product of this new constant tensor and the original tensor, or should we just loop through (in-place), and modify an existing tensor
    - for now, we'll use both but distinguish between the two
    - `_tensor_multiply_by_scalar` will create a new tensor (use const for argument types)
    - `_tensor_multiply_existing_by_scalar` will mutate an existing tensor (void return)
    - see https://softwareengineering.stackexchange.com/questions/422786/naming-convention-for-functions-that-mutate-arguments-vs-creating-a-new-object
- note that reference counting for maintaining topological sort in grad meta is consistent with a variable appearing multiple times in a list of arguments. this is true because `grad_meta_t->ref_count` counts the number of instances in which the variable shows up as an argument, counted by the multiplicity of the argument, rather than just number of graph nodes it's present as an argument in
- currently, coral only supports up to three dimensional tensors
    - will possible be extended to arbitrarily many dimensions (performance considerations)
- too much indirection

AUTOGRAD:
- grad_node_t
    - function pointer to grad function
    - num_parents
    - pointers to parents (two should be fine)
    - pointer to variable
- each "atomic" function comes with a grad version of the function
    - takes in gradient of output (variable_t)
    - every function is either atomic, or can be constructed as the composition of some number of atomic functions
    - sometimes, there is some redundancy in which functions are given grads, for example, both sum, divide, and mean have grads, although, as mean is the composition of sum and divide, it does not need its own grad
        - in general, it helps to provide as many functions as possible with dedicated grad functions, as compositions are expensive, and should be left for arbitrary user-defined functions
    - this simplifies the computation graph


PERFORMANCE CONSIDERATIONS:
- most gradient functions won't actually be inlined

OOP Conventions:
- interface functions always `[MODULE_NAME]_[FUNCTION_NAME]`
- module-internal (static) functions always `[FUNCTION_NAME]`
- constructors: `[MODULE_NAME]_specifics`
    - can no longer have variable called eg `new_tensor`, must be _new_tensor

