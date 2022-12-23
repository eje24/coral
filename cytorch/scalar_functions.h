#ifndef SCALAR_FUNC_H
#define SCALAR_FUNC_H

#include <tensor.h>

typedef tensor_entry_t (* const tensor_scalar_fn_t)(tensor_entry_t left_scalar, tensor_entry_t right_scalar);

static inline tensor_entry_t _scalar_add(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar + right_scalar;
}

static inline tensor_entry_t _scalar_multiply(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar * right_scalar;
}

static inline tensor_entry_t _scalar_subtract(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar - right_scalar;
}


#endif SCALAR_FUNC_H