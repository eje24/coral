# cytorch

A PyTorch-inspired deep learning framework in C.

TODO:
- allow for broadcasting of different but compatible sizes (similar to numpy)
- be consistent about using tensor vs tensor* for static inlined tensor functions (which is better?)
    - for now, pass by value (tensors are small, only stores pointer to data)

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
- apparently cytorch is already taken (it also sounds weird). possible other names:
    - breeze
    - c-torch (concise but uninspiring, current best)
    - libtorch (apparently already a thing: https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-1-why-libtorch/)
    - some element or chemical classification (eg Neon, Halide (taken), etc)
          - tbd
