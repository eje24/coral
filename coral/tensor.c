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

tensor_t* tensor_new_like_with_value(tensor_t* tensor, tensor_entry_t value){
    tensor_t* new_tensor = tensor_new_like(tensor);
    tensor_set_to_scalar_value(new_tensor, value);
    return new_tensor;
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

void tensor_in_place_view_as_shape(tensor_t* tensor, shape_t* new_shape){
    NDEBUG_ASSERT(new_shape->size == tensor->shape->size, "Tensor cannot be viewed in that shape!\n");
    tensor->shape = shape_copy(new_shape);
}

// creates new tensor with desired shape pointing to the same underlying data
tensor_t* tensor_view_as_shape(tensor_t* tensor, shape_t* new_shape){
    NDEBUG_ASSERT(new_shape->size == tensor->shape->size, "Tensor cannot be viewed in that shape!\n");
    tensor_t* new_tensor = (tensor_t*) malloc(sizeof(tensor_t));
    new_tensor->data = tensor->data;
    new_tensor->shape = shape_copy(new_shape);
    return new_tensor;
}

/**
 * COMPARATORS
*/

bool tensor_equal(tensor_t* left_tensor, tensor_t* right_tensor){
    if(!shape_equal(left_tensor->shape, right_tensor->shape)){
        return 0;
    }
    int cmp = memcmp(left_tensor->data, right_tensor->data, tensor_get_size_in_bytes(left_tensor)); 
    return (cmp == 0);
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
    for(size_t dim0_index = 0; dim0_index < tensor->shape->dims[0]; dim0_index++){
        printf("%f ", tensor->data[dim0_index]);
    }
    printf("\n\n");
}

void tensor_display_dim_2(tensor_t* tensor){
    for(size_t dim0_index = 0; dim0_index < tensor->shape->dims[0]; dim0_index++){
        for(size_t dim1_index = 0; dim1_index < tensor->shape->dims[1]; dim1_index++){
            size_t index = dim0_index * tensor->shape->dims[1] + dim1_index;
            printf("%f ", tensor->data[index]);
        }
        printf("\n");
    } 
    printf("\n");
}

void tensor_display_dim_3(tensor_t* tensor){
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
    printf("\n");
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
        default:
            NDEBUG(0, "Display not supported for tensors of dimension greater than three!");
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

static inline tensor_entry_t tensor_entry_divide(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    NDEBUG_ASSERT(right_entry != 0, "Cannot divide by zero!");
    return left_entry / right_entry;
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
    size_t dest_offset_diff = (dest_tensor->shape->dims[dim_index] > 1) ? dest_tensor->shape->strides[dim_index] : 0; 
    size_t dim_length = MAX(source_tensor1->shape->dims[dim_index], source_tensor2->shape->dims[dim_index]);
    for(size_t index = 0; index < dim_length; index++){
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
    size_t dest_offset_diff = (dest_tensor->shape->dims[dim_index] > 1) ? dest_tensor->shape->strides[dim_index] : 0;
    size_t dim_length = MAX(source_tensor1->shape->dims[dim_index], source_tensor2->shape->dims[dim_index]);
    for(size_t index = 0; index < dim_length; index++){
        recursive_in_place_broadcast_fn(dest_tensor, source_tensor1, source_tensor2, dest_offset + index * dest_offset_diff, offset1 + index * offset1_diff, offset2 + index * offset2_diff, dim_index + 1, tensor_entry_binary_fn);
    }
}

void in_place_broadcast_fn(tensor_t* dest_tensor, tensor_t* source_tensor1, tensor_t* source_tensor2, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    NDEBUG_ASSERT(dest_tensor != source_tensor1 && dest_tensor != source_tensor2, "Destination and source tensors cannot alias the same memory - undefined behavior!");
    NDEBUG_ASSERT(shape_equal(shape_get_broadcast_shape(source_tensor1->shape, source_tensor2->shape), dest_tensor->shape), "Destination tensor has improper shape!");
    shape_display(source_tensor1->shape);
    shape_display(source_tensor2->shape);
    NDEBUG_ASSERT(tensor_broadcast_compatible(source_tensor1, source_tensor2), "Tensors are not broadcast compatible!\n");
    int source_dims1 = TENSOR_NUM_DIMS(source_tensor1);
    int source_dims2 = TENSOR_NUM_DIMS(source_tensor2);
    // pad the smaller tensor with leading dimensions of length 1 so that both tensors have the same number of dimensions
    if(source_dims1 < source_dims2){
        source_tensor1 = tensor_view_as_shape(source_tensor1, shape_extend_to_dims(source_tensor1->shape, source_dims2));
    }else if(source_dims1 > source_dims2){
        source_tensor2 = tensor_view_as_shape(source_tensor2, shape_extend_to_dims(source_tensor2->shape, source_dims1));
    }
    recursive_in_place_broadcast_fn(dest_tensor, source_tensor1, source_tensor2, 0, 0, 0, 0, tensor_entry_binary_fn);
}

// TODO : allow for broadcasting of different sizes
tensor_t* tensor_broadcast_fn(tensor_t* left_tensor, tensor_t* right_tensor, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    // ensure that left_tensor->num_dims >= right_tensor->num_dims
    tensor_t* new_tensor = tensor_new(shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape));
    in_place_broadcast_fn(new_tensor, left_tensor, right_tensor, tensor_entry_binary_fn);
    return new_tensor;
}

/**
 * sums along a subset of the dimensions so that the resulting tensor has shape target_shape
*/
tensor_t* tensor_reduce_to_shape(tensor_t* tensor, shape_t* target_shape){
    NDEBUG_ASSERT(shape_broadcast_compatible(tensor->shape, target_shape), "Tensor is not compatible with target shape.");
    NDEBUG_ASSERT(target_shape->num_dims <= TENSOR_NUM_DIMS(tensor), "Target shape has too many dimensions.");
    shape_t* extended_target_shape = shape_extend_to_dims(target_shape, TENSOR_NUM_DIMS(tensor));
    tensor_t* reduced_tensor = tensor_new(extended_target_shape);
    recursive_in_place_broadcast_fn(reduced_tensor, reduced_tensor, tensor, 0, 0, 0, 0, &tensor_entry_add);
    return tensor_view_as_shape(reduced_tensor, target_shape);
}

/**
 * MUTATING FUNCTIONS
 * tensor_in_place_op(left_tensor, right_tensor) sets
 * left_tensor <- op(left_tensor, right_tensor)
*/


/**
 * Adds right_tensor to left_tensor
 */
void tensor_in_place_add(tensor_t* left_tensor, tensor_t* right_tensor){
    NDEBUG_ASSERT(shape_equal(shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape), left_tensor->shape), "Left tensor has improper shape for in place operation!");
    left_tensor->data = tensor_add(left_tensor, right_tensor)->data;
}

void tensor_in_place_subtract(tensor_t* left_tensor, tensor_t* right_tensor){
    NDEBUG_ASSERT(shape_equal(shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape), left_tensor->shape), "Left tensor has improper shape for in place operation!");
    left_tensor->data = tensor_subtract(left_tensor, right_tensor)->data;
}

void tensor_in_place_multiply(tensor_t* left_tensor, tensor_t* right_tensor){
    NDEBUG_ASSERT(shape_equal(shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape), left_tensor->shape), "Left tensor has improper shape for in place operation!");
    left_tensor = tensor_multiply(left_tensor, right_tensor);
}

void tensor_in_place_divide(tensor_t* left_tensor, tensor_t* right_tensor){
    NDEBUG_ASSERT(shape_equal(shape_get_broadcast_shape(left_tensor->shape, right_tensor->shape), left_tensor->shape), "Left tensor has improper shape for in place operation!");
    left_tensor = tensor_divide(left_tensor, right_tensor);
}

void tensor_in_place_multiply_by_scalar(tensor_t* tensor, tensor_entry_t value){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = tensor_entry_multiply(tensor_get_entry(tensor, index), value);
        tensor_set_entry(tensor, index, new_value);
    }
}

void tensor_in_place_divide_by_scalar(tensor_t* tensor, tensor_entry_t value){
    size_t tensor_size = tensor_get_size(tensor);
    for(size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = tensor_entry_divide(tensor_get_entry(tensor, index), value);
        tensor_set_entry(tensor, index, new_value);
    }
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

tensor_t* tensor_divide(tensor_t* left_tensor, tensor_t* right_tensor){
    return tensor_broadcast_fn(left_tensor, right_tensor, &tensor_entry_divide);
}

tensor_t* tensor_multiply_by_scalar_grad(tensor_t* tensor, tensor_entry_t value){
    return tensor_new_like_with_value(tensor, value);
}

tensor_t* tensor_multiply_by_scalar(tensor_t* tensor, tensor_entry_t value){
    tensor_t* new_tensor = tensor_copy(tensor);
    tensor_in_place_multiply_by_scalar(new_tensor, value);
    return new_tensor;
}

tensor_t* tensor_divide_by_scalar_grad(tensor_t* tensor, tensor_entry_t value){
    NDEBUG_ASSERT(value != 0, "Cannot divide by zero!");
    return tensor_new_like_with_value(tensor, 1 / value);
}

tensor_t* tensor_divide_by_scalar(tensor_t* tensor, tensor_entry_t value){
    tensor_t* new_tensor = tensor_copy(tensor);
    tensor_in_place_divide_by_scalar(new_tensor, value);
    return new_tensor;
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
    return tensor_new_like_with_value(tensor, 1.0);
}

tensor_t* tensor_sum(tensor_t* tensor){
    size_t tensor_size = tensor_get_size(tensor);
    tensor_entry_t sum = 0;
    for(size_t index = 0; index < tensor_size; index++){
        sum += tensor_get_entry(tensor, index);
    }
    return tensor_new_from_entry(sum);
}

tensor_t* tensor_mean_grad(tensor_t* tensor){
    return tensor_divide_by_scalar(tensor_new_like_with_value(tensor, 1), tensor_get_size(tensor));
}

tensor_t* tensor_mean(tensor_t* tensor){
    NDEBUG_ASSERT(tensor_get_size(tensor), "Cannot take mean of tensor of size zero!");
    return tensor_divide_by_scalar(tensor_sum(tensor), tensor_get_size(tensor));
}

/**
 * add tensor divide (with appropriate checks
 * add tensor divide in place )
*/
