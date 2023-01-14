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
typedef tensor_entry_t (* tensor_entry_binary_fn_t)(tensor_entry_t left_entry, tensor_entry_t right_entry);
// entry -> entry, applied component wise to create a new tensor from two existing ones
typedef tensor_entry_t (* tensor_entry_unary_fn_t)(tensor_entry_t entry);
// binary op: index: size_t -> entry: tensor_entry_t, used to populate a tensor
typedef tensor_entry_t (* tensor_index_fn_t)(size_t index);

tensor_t* tensor_new(shape_t* shape);
tensor_t* tensor_new_like(tensor_t* old_tensor);
tensor_t* tensor_new_zeros_like(tensor_t* old_tensor);
tensor_t* tensor_copy(tensor_t* old_tensor);
tensor_t* tensor_view_as(tensor_t* tensor, shape_t* new_shape);

bool tensor_equal(tensor_t* left_tensor, tensor_t* right_tensor);

bool tensor_is_scalar(tensor_t* tensor);

void tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value);
void tensor_in_place_apply_index_fn(tensor_t* tensor, tensor_index_fn_t index_fn);
void tensor_in_place_apply_entry_fn(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn);

/**
 * TENSOR_INDEX_FN_T's, TENSOR_ENTRY_FN_T's
*/

static inline tensor_entry_t index_identity(size_t index){
    return index;
}

/**
 * GETTERS AND SETTERS
 * NOTE: in .h so they'll be inlined
*/

static inline tensor_entry_t tensor_get_entry(tensor_t* tensor, size_t index){
    // DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    return tensor->data[index];
}

static inline void tensor_set_entry(tensor_t* tensor, size_t index, tensor_entry_t value){
    // DEBUG_ASSERT(!TENSOR_IN_BOUNDS_INDEX(tensor, index), "Out of bounds!\n");
    tensor->data[index] = value;
}

void tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value);
void tensor_set_to_fn_value(tensor_t* tensor, tensor_index_fn_t index_fn);


void tensor_display(tensor_t* tensor);

void tensor_in_place_add(tensor_t* left_tensor, tensor_t* right_tensor);
void tensor_in_place_subtract(tensor_t* left_tensor, tensor_t* right_tensor);
void tensor_in_place_multiply(tensor_t* left_tensor, tensor_t* right_tensor);
void tensor_in_place_multiply_by_scalar(tensor_t* tensor, tensor_entry_t value);
void tensor_in_place_divide_by_scalar(tensor_t* tensor, tensor_entry_t value);


tensor_t* tensor_add(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* tensor_subtract(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* tensor_multiply(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* tensor_divide(tensor_t* left_tensor, tensor_t* right_tensor);
tensor_t* tensor_multiply_by_scalar(tensor_t* tensor, tensor_entry_t value);
tensor_t* tensor_divide_by_scalar(tensor_t* tensor, tensor_entry_t value);
tensor_t* tensor_abs_grad(tensor_t* tensor);
tensor_t* tensor_abs(tensor_t* tensor);
tensor_t* tensor_sum_grad(tensor_t* tensor);
tensor_t* tensor_sum(tensor_t* tensor);

#endif // TENSOR_H