#ifndef BASIC_FUNC_H
#define BASIC_FUNC_H

#include "tensor.h"
#include <stdbool.h>

// returns true if and only if tensor dimensions match exactly
// used as a pre-check for component-wise operations
static inline bool dimensions_match(tensor_t left_tensor, tensor_t right_tensor){
    return (left_tensor.num_columns == right_tensor.num_columns) && (left_tensor.num_rows == right_tensor.num_rows);
}

// returns true if and only if internal dimensions match
// so that left_tensor @ right_tensor is a valid matrix multiplication
static inline bool internal_dimensions_match(tensor_t left_tensor, tensor_t right_tensor){
    return (left_tensor.num_columns == right_tensor.num_rows);
}

#endif // BASIC_FUNC_H
