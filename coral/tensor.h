#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "assert.h"
#include "shape.h"

typedef float tensor_entry_t; 

#define TENSOR_MAX_DIMS 3

typedef struct {
    tensor_entry_t* data; // ptr to data
    shape_t* shape; //dimensions of data
} tensor_t;

// macros for debugging
// note that lower bound check is currently redundant as size_t is uint64_t
#define TENSOR_NUM_DIMS(tensor) (tensor)->shape->num_dims
#define TENSOR_IN_BOUNDS_INDEX(tensor, index) (0 <= (index) && (index) < tensor->num_rows * tensor->num_columns)

// entry x entry -> entry, applied component wise to create a new tensor from two existing ones
typedef tensor_entry_t (* const tensor_entry_binary_fn_t)(tensor_entry_t left_entry, tensor_entry_t right_entry);
// entry -> entry, applied component wise to create a new tensor from two existing ones
typedef tensor_entry_t (* const tensor_entry_unary_fn_t)(tensor_entry_t entry);
// binary op: index: size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* const tensor_index_fn_t)(size_t index);

tensor_t* _new_tensor(shape_t* shape);
tensor_t* _new_tensor_like(const tensor_t* old_tensor);
tensor_t* _new_tensor_zeros_like(const tensor_t* old_tensor);
tensor_t* _copy_tensor(const tensor_t* old_tensor);

bool _tensor_is_scalar(tensor_t* tensor);

void _tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value);
void _tensor_multiply_by_scalar_value(tensor_t* tensor, tensor_entry_t value);
void _tensor_multiply_existing(tensor_t* multiplicand, tensor_t* multiplier);
void _tensor_set_to_index_fn_value(tensor_t* tensor, tensor_index_fn_t index_fn);
void _tensor_set_to_entry_fn_value(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn);

/**
 * GETTERS AND SETTERS
 * NOTE: in .h so they'll be inlined
*/

static inline tensor_entry_t _tensor_get_entry(const tensor_t* tensor, size_t index){
    // DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    return tensor->data[index];
}

static inline void _tensor_set_entry(const tensor_t* tensor, size_t index, tensor_entry_t value){
    // DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    tensor->data[index] = value;
}

void _tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value);
void _tensor_set_to_fn_value(tensor_t* tensor, tensor_index_fn_t index_fn);


void _display_tensor(tensor_t* tensor);

void _tensor_add_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor);
void _tensor_subtract_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor);
void _tensor_multiply_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor);

tensor_t* _tensor_add(const tensor_t* left_tensor, const tensor_t* right_tensor);
tensor_t* _tensor_subtract(const tensor_t* left_tensor, const tensor_t* right_tensor);
tensor_t* _tensor_multiply(const tensor_t* left_tensor, const tensor_t* right_tensor);
tensor_t* _tensor_abs_grad(const tensor_t* tensor);
tensor_t* _tensor_abs(const tensor_t* tensor);
tensor_t* _tensor_sum_grad(const tensor_t* tensor);
tensor_t* _tensor_sum(const tensor_t* tensor);

#endif // TENSOR_H