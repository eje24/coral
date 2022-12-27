# coral

A PyTorch-inspired deep learning framework in C.

TODO:
- misc
    - ✅ change name (DONE)
- tensor
    - 🏗️ assert that dimenions are correct/compatible when doing operations
    - 🏗️ start test suite
    - 🏗️ add struct constant_t, and make variable_t an extension
    - 🏗️ enable link-time optimization (quick)
    - extend tensor index/entry value lambda broadcasts to variable
    - 🏗️ add in ability to construct different views of the same tensor (just have another tensor pointing to the same data, but with different num_rows, num_columns)
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