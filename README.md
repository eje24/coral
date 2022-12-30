# coral

A PyTorch-inspired deep learning framework in C.

TODO:
- misc
    - ✅ change name (DONE)
- tensor
    - 🏗️ assert that dimenions are correct/compatible when doing operations
    - 🏗️ functions which do not modify should have const arguments (_tensor_add, _tensor_subtract, etc)
    - 🏗️ update _tensor_broadcast_scalar_fn to two versions (binary and unary) (current implementation is binary)
    - 🏗️ start test suite
    - 🏗️ add struct constant_t, and make variable_t an extension
    - 🏗️ enable link-time optimization (quick)
    - extend tensor index/entry value lambda broadcasts to variable
    - 🏗️ add in ability to construct different views of the same tensor (just have another tensor pointing to the same data, but with different num_rows, num_columns)
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
- function/operator overloading would be useful
- union types aren't great
- how should we multiply tensors by a constant? should we construct a new tensor with all entries equal to that constant and then return the product of this new constant tensor and the original tensor, or should we just loop through (in-place), and modify an existing tensor
    - for now, we'll use both but distinguish between the two
    - `_tensor_multiply_by_scalar` will create a new tensor (use const for argument types)
    - `_tensor_multiply_existing_by_scalar` will mutate an existing tensor (void return)
    - see https://softwareengineering.stackexchange.com/questions/422786/naming-convention-for-functions-that-mutate-arguments-vs-creating-a-new-object

AUTOGRAD:
- grad_node_t
    - function pointer to grad function
    - num_parents
    - pointers to parents (two should be fine)
    - pointer to variable
- each function comes with a grad version of the function
    - takes in gradient of output (variable_t)

