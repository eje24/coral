# cytorch

A PyTorch-inspired deep-learning framework in C.

Drawbacks:
- no generic programming: hard to easily create tensors for different types. for now we'll just stick to floats
- sizes are two dimensional (hard to make variable number of arguments

TODO:
- allow for broadcasting of different but compatible sizes (similar to numpy)
- be consistent about using tensor vs tensor* for static inlined tensor functions (which is better?)
    - for now, pass by value (tensors are small, only stores pointer to data)

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
- apparently cytorch is already taken (it also sounds weird). possible other names:
    - breeze
    - c-torch (concise)
    - libtorch (apparently already a thing: https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-1-why-libtorch/)
