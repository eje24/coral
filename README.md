# cytorch

Drawbacks:
- no generic programming: hard to easily create tensors for different types. for now we'll just stick to floats
- sizes are two dimensional (hard to make variable number of arguments

TODO:
- allow for broadcasting of different but compatible sizes (similar to numpy)
- be consistent about using tensor vs tensor* for static inlined tensor functions (which is better?)
    - for now, pass by value (tensors are small, only stores pointer to data)
    - for future: pass by reference for added flexibility (if we add more fields to tensor later)
- buildout test suite
- add in ability to populate tensor with function
- add in ability to construct different views of the same tensor (just have another tensor pointing to the same data, but with different num_rows, num_columns)

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
