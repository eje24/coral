# grad

A PyTorch-inspired deep learning framework in C.

TODO:
- change name
- add link-time optimization (quick)
- allow for broadcasting of different but compatible sizes (similar to numpy)
- be consistent about using tensor vs tensor* for static inlined tensor functions (which is better?)
    - for now, pass by value (tensors are small, only stores pointer to data)
    - for future: pass by reference for added flexibility (if we add more fields to tensor later)
- buildout test suite
- add in ability to populate tensor with function
- add in ability to construct different views of the same tensor (just have another tensor pointing to the same data, but with different num_rows, num_columns)

NOTES:
- static inlining vs macro for "short" tensor functions?? current approach is to use macros for short debugging functions (eg in-bounds checks)
- apparently cytorch is already taken (it also sounds weird). possible other names:
    - breeze
    - c-torch (concise but uninspiring, current best)
    - libtorch (apparently already a thing: https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-1-why-libtorch/)
    - some element or chemical classification (eg Neon, Halide (taken), etc)
          - tbd
