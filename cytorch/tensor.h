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

// entry x entry -> entry, applied component wise to create a new tensor from two existing ones
typedef tensor_entry_t (* const tensor_entry_fn_t)(tensor_entry_t left_entry, tensor_entry_t right_entry);
// binary op: index: tensor_size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* const tensor_tensor_fn_t)(tensor_size_t row, tensor_size_t index);
// binary op: row: tensor_size_t x column: tensor_size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* const tensor_row_column_fn_t)(tensor_size_t row, tensor_size_t column);


tensor_t _new_tensor(tensor_size_t num_rows, tensor_size_t num_columns);
tensor_t _new_tensor_like(tensor_t old_tensor);
tensor_t _new_tensor_zeros_like(tensor_t old_tensor);
tensor_t _copy_tensor(tensor_t old_tensor);
void _populate_tensor(tensor_t* tensor, tensor_row_column_fn_t row_column_fn);

void display_tensor(tensor_t tensor);

tensor_t _tensor_add(tensor_t left_tensor, tensor_t right_tensor);
tensor_t _tensor_subtract(tensor_t left_tensor, tensor_t right_tensor);
tensor_t _tensor_multiply(tensor_t left_tensor, tensor_t right_tensor);

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