#include "tensor.h"
#include "utils.h"
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// NOTE: for now using static inline over macro for type safety
// macro doesn't feel right here

static inline size_t tensor_get_size(tensor_t* tensor){
    return tensor->shape->size;
}

// returns size of broadcast
// static inline shape_t* tensor_get_broadcast_shape(tensor_t* left_tensor, tensor_t* right_tensor){
//     return shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape);
// }

static inline size_t tensor_get_size_in_bytes(tensor_t* tensor){
    return tensor_get_size(tensor) * sizeof(tensor_entry_t);
}

// create new tensor
// entries are set to zero by default
tensor_t* tensor_new(shape_t* shape){
    tensor_entry_t* data = (tensor_entry_t*) calloc(shape->size, sizeof(tensor_entry_t));
    tensor_t* new_tensor = (tensor_t*) malloc(sizeof(tensor_t));
    new_tensor->data = data;
    new_tensor->shape = shape_copy(shape);
    return new_tensor;
}

// TODO - ensure this is inlined
tensor_t* tensor_new_like(tensor_t* tensor){
    return tensor_new(tensor->shape);
}

// the same as tensor_new_like (for now)
tensor_t* tensor_new_zeros_like(tensor_t* tensor){
    return tensor_new(tensor->shape);
}

tensor_t* tensor_new_from_entry(tensor_entry_t entry){
    size_t dims = 1;
    tensor_t* new_tensor = tensor_new(shape_new(1, &dims));
    tensor_set_entry(new_tensor, 0, entry);
    return new_tensor;
}


tensor_t* tensor_copy(tensor_t* old_tensor){
    tensor_t* new_tensor = tensor_new_like(old_tensor);
    memcpy(new_tensor->data, old_tensor->data, tensor_get_size_in_bytes(old_tensor));
    return new_tensor;
}

tensor_t* tensor_view_as(tensor_t* tensor, shape_t* new_shape){
    NDEBUG_ASSERT(new_shape->size == tensor->shape->size, "Tensor cannot be viewed in that shape!\n");
    tensor_t* new_tensor = tensor_copy(tensor);
    new_tensor->shape = shape_copy(new_shape);
    return new_tensor;
}


/**
 * PROPERTIES
*/

// returns True iff the tensors dimensions are 1
bool tensor_is_scalar(tensor_t* tensor){
    return shape_is_scalar(tensor->shape);
}

/**
 * NON-INLINED SETTERS/MUTATORS
*/

void tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_set_entry(tensor, index, value);
    }
}

void tensor_in_place_multiply_by_scalar_value(tensor_t* tensor, tensor_entry_t value){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = value * tensor_get_entry(tensor, index);
        tensor_set_entry(tensor, index, new_value);
    }
}

void tensor_in_place_multiply(tensor_t* multiplicand, tensor_t* multiplier){
    size_t tensor_size = tensor_get_size(multiplicand);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = tensor_get_entry(multiplicand, index) * tensor_get_entry(multiplier, index);
        tensor_set_entry(multiplicand, index, new_value);
    }
}

void tensor_in_place_apply_index_fn(tensor_t* tensor, tensor_index_fn_t index_fn){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t entry_value = (*index_fn)(index);
        tensor_set_entry(tensor, index, entry_value);
    }
}

void tensor_in_place_apply_entry_fn(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t entry_old_value = tensor_get_entry(tensor, index);
        tensor_entry_t entry_new_value = (*entry_fn)(entry_old_value);
        tensor_set_entry(tensor, index, entry_new_value);
    }
}

// static inline tensor_t* tensor_apply_index_fn(tensor_t* tensor, tensor_index_fn_t index_fn){
//     tensor_t* new_tensor = tensor_copy(tensor);
//     tensor_in_place_apply_index_fn(new_tensor, index_fn);
//     return new_tensor;
// }

// static inline tensor_t* tensor_apply_entry_fn(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn){
//     tensor_t* new_tensor = tensor_copy(tensor);
//     tensor_in_place_apply_entry_fn(new_tensor, entry_fn);
//     return new_tensor;
// }




/**
 * PRINTING
*/

void tensor_display_dim_1(tensor_t* tensor){
    printf("Dimensions: %zu\n", tensor->shape->dims[0]);
    for(size_t dim0_index = 0; dim0_index < tensor->shape->dims[0]; dim0_index++){
        printf("%f ", tensor->data[dim0_index]);
    }
    printf("\n");
}

void tensor_display_dim_2(tensor_t* tensor){
    printf("Dimensions: %zu x %zu\n", tensor->shape->dims[0], tensor->shape->dims[1]);
    for(size_t dim0_index = 0; dim0_index < tensor->shape->dims[0]; dim0_index++){
        for(size_t dim1_index = 0; dim1_index < tensor->shape->dims[1]; dim1_index++){
            size_t index = dim0_index * tensor->shape->dims[1] + dim1_index;
            printf("%f ", tensor->data[index]);
        }
        printf("\n");
    } 
}

void tensor_display_dim_3(tensor_t* tensor){
    printf("Dimensions: %zu x %zu x %zu\n", tensor->shape->dims[0], tensor->shape->dims[1], tensor->shape->dims[2]);
    for(size_t dim0_index = 0; dim0_index < tensor->shape->dims[0]; dim0_index++){
        for(size_t dim1_index = 0; dim1_index < tensor->shape->dims[1]; dim1_index++){
            for(size_t dim2_index = 0; dim2_index < tensor->shape->dims[2]; dim2_index++){
                size_t index = dim0_index * tensor->shape->dims[1] * tensor->shape->dims[2] + dim1_index * tensor->shape->dims[2] + dim2_index;
                printf("%f ", tensor->data[index]);
            }
            printf("\n");
        }
        printf("\n");
    } 
}

void tensor_display(tensor_t* tensor){
    shape_display(tensor->shape);
    switch(TENSOR_NUM_DIMS(tensor)){
        case 1:
            tensor_display_dim_1(tensor);
            break;
        case 2:
            tensor_display_dim_2(tensor);
            break;
        case 3:
            tensor_display_dim_3(tensor);
            break;
    }
}

/**
 * VIEWS - TODO
*/

// returns true if and only if tensor dimensions match exactly
// used as a pre-check for component-wise operations
// bool tensor_exact_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
//     if(left_tensor->num_dims != left_tensor->num_dims){
//         return 0;
//     }
//     for(int dim_index = 0; dim_index < left_tensor->num_dims; dim_index++){
//         if(left_tensor->dims[dim_index] != right_tensor->dims[dim_index]){
//             return 0;
//         }
//     }
//     return 1;
// }

// returns true iff tensors can be broadcasted in the same manner as in numpy/PyTorch
bool tensor_broadcast_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
    return shape_broadcast_compatible(left_tensor->shape, right_tensor->shape);
}

/**
 * tensor_entry_t x tensor_entry_t -> tensor_entry_t
 * tensor_entry_t -> tensor_entry_t
*/

static inline tensor_entry_t tensor_entry_add(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry + right_entry;
}

static inline tensor_entry_t tensor_entry_multiply(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry * right_entry;
}

static inline tensor_entry_t tensor_entry_subtract(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry - right_entry;
}

static inline tensor_entry_t tensor_entry_abs_grad(tensor_entry_t entry){
    return entry >= 0 ? 1 : -1;
}

static inline tensor_entry_t tensor_entry_abs(tensor_entry_t entry){
    return entry >= 0 ? entry : -entry;
}

// called when dim_index + 1 = dest_tensor->shape->num_dims
void base_in_place_broadcast(tensor_t* dest_tensor, tensor_t* source_tensor1, tensor_t* source_tensor2, size_t dest_offset, size_t offset1, size_t offset2, int dim_index, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    size_t offset1_diff = (source_tensor1->shape->dims[dim_index] > 1) ? source_tensor1->shape->strides[dim_index] : 0;
    size_t offset2_diff = (source_tensor2->shape->dims[dim_index] > 1) ? source_tensor2->shape->strides[dim_index] : 0;
    size_t dest_offset_diff = dest_tensor->shape->strides[dim_index]; 
    for(size_t index = 0; index < dest_tensor->shape->dims[dim_index]; index++){
        tensor_entry_t source_entry1 = tensor_get_entry(source_tensor1, offset1 + index * offset1_diff);
        tensor_entry_t source_entry2 = tensor_get_entry(source_tensor2, offset2 + index * offset2_diff);
        tensor_entry_t new_entry = (*tensor_entry_binary_fn)(source_entry1, source_entry2);
        tensor_set_entry(dest_tensor, dest_offset + index * dest_offset_diff, new_entry);
    }    
    return;
}

void recursive_in_place_broadcast_fn(tensor_t* dest_tensor, tensor_t* source_tensor1, tensor_t* source_tensor2, size_t dest_offset, size_t offset1, size_t offset2, int dim_index, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    // base case
    if(dim_index + 1 == TENSOR_NUM_DIMS(dest_tensor)){
        base_in_place_broadcast(dest_tensor, source_tensor1, source_tensor2, dest_offset, offset1, offset2, dim_index, tensor_entry_binary_fn);
        return;
    }
    size_t offset1_diff = (source_tensor1->shape->dims[dim_index] > 1) ? source_tensor1->shape->strides[dim_index] : 0;
    size_t offset2_diff = (source_tensor2->shape->dims[dim_index] > 1) ? source_tensor2->shape->strides[dim_index] : 0;
    size_t dest_offset_diff = dest_tensor->shape->strides[dim_index];
    for(size_t index = 0; index < dest_tensor->shape->dims[dim_index]; index++){
        recursive_in_place_broadcast_fn(dest_tensor, source_tensor1, source_tensor2, dest_offset + index * dest_offset_diff, offset1 + index * offset1_diff, offset2 + index * offset2_diff, dim_index + 1, tensor_entry_binary_fn);
    }
}

void in_place_broadcast_fn(tensor_t* dest_tensor, tensor_t* source_tensor1, tensor_t* source_tensor2, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    NDEBUG_ASSERT(tensor_broadcast_compatible(source_tensor1, source_tensor2), "Tensors are not broadcast compatible!\n");
    int source_dims1 = TENSOR_NUM_DIMS(source_tensor1);
    int source_dims2 = TENSOR_NUM_DIMS(source_tensor2);
    // pad the smaller tensor with leading dimensions of length 1 so that both tensors have the same number of dimensions
    if(source_dims1 < source_dims2){
        source_tensor1 = tensor_view_as(source_tensor1, shape_extend_to_dims(source_tensor1->shape, source_dims2));
    }else if(source_dims1 > source_dims2){
        source_tensor2 = tensor_view_as(source_tensor2, shape_extend_to_dims(source_tensor2->shape, source_dims1));
    }
    recursive_in_place_broadcast_fn(dest_tensor, source_tensor1, source_tensor2, 0, 0, 0, 0, tensor_entry_binary_fn);
}

// TODO : allow for broadcasting of different sizes
tensor_t* tensor_broadcast_fn(tensor_t* left_tensor, tensor_t* right_tensor, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    // ensure that left_tensor->num_dims >= right_tensor->num_dims
    tensor_t* new_tensor = (TENSOR_NUM_DIMS(left_tensor) > TENSOR_NUM_DIMS(right_tensor)) ? tensor_new_like(left_tensor) : tensor_new_like(right_tensor);
    in_place_broadcast_fn(new_tensor, left_tensor, right_tensor, tensor_entry_binary_fn);
    return new_tensor;
}

/**
 * MUTATING FUNCTIONS
*/

void tensor_add_to_existing(tensor_t* left_tensor, tensor_t* right_tensor){
    in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &tensor_entry_add);
}

void tensor_subtract_to_existing(tensor_t* left_tensor, tensor_t* right_tensor){
    in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &tensor_entry_subtract);
}

void tensor_multiply_to_existing(tensor_t* left_tensor, tensor_t* right_tensor){
    in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &tensor_entry_multiply);
}

/**
 *  NON-MUTATING FUNCTIONS
*/

// return new tensor which is the result of component-wise addition 
// of left_tensor and right_tensor
// assumes that left_tensor and right_tensor are compatible
tensor_t* tensor_add(tensor_t* left_tensor, tensor_t* right_tensor){
    return tensor_broadcast_fn(left_tensor, right_tensor, &tensor_entry_add);
}

tensor_t* tensor_subtract(tensor_t* left_tensor, tensor_t* right_tensor){
    return tensor_broadcast_fn(left_tensor, right_tensor, &tensor_entry_subtract);
}

tensor_t* tensor_multiply(tensor_t* left_tensor, tensor_t* right_tensor){
    return tensor_broadcast_fn(left_tensor, right_tensor, &tensor_entry_multiply);
}

tensor_t* tensor_abs_grad(tensor_t* tensor){
    tensor_t* grad_tensor = tensor_copy(tensor);
    tensor_in_place_apply_entry_fn(grad_tensor, &tensor_entry_abs_grad);
    return grad_tensor;
}

tensor_t* tensor_abs(tensor_t* tensor){
    tensor_t* new_tensor = tensor_copy(tensor);
    tensor_in_place_apply_entry_fn(new_tensor, &tensor_entry_abs);
    return new_tensor;
}

tensor_t* tensor_sum_grad(tensor_t* tensor){
    tensor_t* grad_tensor = tensor_new_like(tensor);
    tensor_set_to_scalar_value(grad_tensor, 1);
    return grad_tensor;
}

tensor_t* tensor_sum(tensor_t* tensor){
    size_t tensor_size = tensor_get_size(tensor);
    tensor_entry_t sum = 0;
    for(size_t index = 0; index < tensor_size; index++){
        sum += tensor_get_entry(tensor, index);
    }
    return tensor_new_from_entry(sum);
}
