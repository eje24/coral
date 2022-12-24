#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "debug.h"

typedef float tensor_entry_t; 
typedef uint64_t tensor_size_t;

typedef struct {
    tensor_entry_t* data;
    tensor_size_t num_rows;
    tensor_size_t num_columns;
} tensor_t;

// macros for debugging
// note that lower bound check is currently redundant as tensor_size_t is uint64_t
#define TENSOR_IN_BOUNDS_ROW_COLUMN(tensor, row, column) (0 <= (row) && (row) <= tensor->num_rows && 0 <= (column) && (column) <= tensor->num_columns)
#define TENSOR_IN_BOUNDS_INDEX(tensor, index) (0 <= (index) && (index) < tensor->num_rows * tensor->num_columns)

// entry x entry -> entry, applied component wise to create a new tensor from two existing ones
typedef tensor_entry_t (* const tensor_entry_fn_t)(tensor_entry_t left_entry, tensor_entry_t right_entry);
// binary op: index: tensor_size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* const tensor_tensor_fn_t)(tensor_size_t row, tensor_size_t index);
// binary op: row: tensor_size_t x column: tensor_size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* const tensor_row_column_fn_t)(tensor_size_t row, tensor_size_t column);


tensor_t* _new_tensor(tensor_size_t num_rows, tensor_size_t num_columns);
tensor_t* _new_tensor_like(tensor_t* old_tensor);
tensor_t* _new_tensor_zeros_like(tensor_t* old_tensor);
tensor_t* _copy_tensor(tensor_t* old_tensor);
void _populate_tensor(tensor_t* tensor, tensor_row_column_fn_t row_column_fn);

void _display_tensor(tensor_t* tensor);

tensor_t* _tensor_add(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* _tensor_subtract(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* _tensor_multiply(tensor_t* left_tensor, tensor_t* right_tensor);

/**
 * GETTERS AND SETTERS
 * NOTE: in .h so they'll be inlined
*/

static inline tensor_entry_t _tensor_get_entry(tensor_t* tensor, tensor_size_t index){
    DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    return tensor->data[index];
}

static inline tensor_entry_t _tensor_get_entry_row_column(tensor_t* tensor, tensor_size_t row, tensor_size_t column){
    DEBUG_ASSERT(!TENSOR_IN_BOUNDS_ROW_COLUMN(tensor, row, column), "Out of bounds!\n");
    tensor_size_t index = row * tensor->num_columns + column;
    return tensor->data[index];
}

static inline void _tensor_set_entry(tensor_t *tensor, tensor_size_t index, tensor_entry_t value){
    DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    tensor->data[index] = value;
}

static inline void _tensor_set_entry_row_column(tensor_t* tensor, tensor_size_t row, tensor_size_t column, tensor_entry_t value){
    DEBUG_ASSERT(!TENSOR_IN_BOUNDS_ROW_COLUMN(tensor, row, column), "Out of bounds!\n");
    tensor_size_t index = row * tensor->num_columns + column;
    tensor->data[index] = value;
}
#endif // TENSOR_H