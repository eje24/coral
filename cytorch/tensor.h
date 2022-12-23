#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>

typedef float tensor_entry_t; 
typedef uint64_t tensor_size_t;

typedef struct {
    tensor_entry_t* data;
    tensor_size_t num_rows;
    tensor_size_t num_columns;
} tensor_t;

typedef tensor_entry_t (* const tensor_scalar_fn_t)(tensor_entry_t left_scalar, tensor_entry_t right_scalar);

tensor_t _new_tensor(tensor_size_t num_rows, tensor_size_t num_columns);
tensor_t _new_tensor_like(tensor_t old_tensor);
tensor_t _new_tensor_zeros_like(tensor_t old_tensor);
tensor_t _copy_tensor(tensor_t old_tensor);

tensor_t _tensor_add(tensor_t left_tensor, tensor_t right_tensor);
inline uint8_t set_tensor_entry(tensor_t* tensor, tensor_size_t row, tensor_size_t column, tensor_entry_t value);

// returns true if and only if tensor dimensions match exactly
// used as a pre-check for component-wise operations
static inline bool tensor_dimensions_match(tensor_t left_tensor, tensor_t right_tensor){
    return (left_tensor.num_columns == right_tensor.num_columns) && (left_tensor.num_rows == right_tensor.num_rows);
}

// returns true if and only if internal dimensions match
// so that left_tensor @ right_tensor is a valid matrix multiplication
static inline bool tensor_internal_dimensions_match(tensor_t left_tensor, tensor_t right_tensor){
    return (left_tensor.num_columns == right_tensor.num_rows);
}
#endif // TENSOR_H