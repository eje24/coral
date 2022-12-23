# cytorch

Drawbacks:
- no generic programming: hard to easily create tensors for different types. for now we'll just stick to floats
- sizes are two dimensional (hard to make variable number of arguments

TODO:
- allow for broadcasting of different but compatible sizes (similar to numpy)
- be consistent about using tensor vs tensor* for static inlined tensor functions (which is better?)
    - for now, pass by value (tensors are small, only stores pointer to data)

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
