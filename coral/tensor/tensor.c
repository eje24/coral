#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// NOTE: for now using static inline over macro for type safety
// macro doesn't feel right here

static inline tensor_size_t _get_size_from_dims(uint8_t num_dims, tensor_size_t* dims){
    tensor_size_t size = 1;
    for(uint8_t dim_idx = 0; dim_idx < num_dims; dim_idx++){
        size *= dims[dim_idx];
    }
    return size;
}

static inline tensor_size_t _tensor_get_size(const tensor_t* tensor){
    return _get_size_from_dims(tensor->num_dims, tensor->dims);
}

static inline size_t _tensor_get_size_in_bytes(const tensor_t* tensor){
    return _tensor_get_size(tensor) * sizeof(tensor_entry_t);
}

// create new tensor
// entries are set to zero by default
tensor_t* _new_tensor(uint8_t num_dims, const tensor_size_t* dims){
    tensor_size_t tensor_size = _get_size_from_dims(num_dims, dims);
    tensor_entry_t* data = (tensor_entry_t*) calloc(tensor_size, sizeof(tensor_entry_t));
    tensor_t* new_tensor = (tensor_t*) malloc(sizeof(tensor_t));
    new_tensor->data = data;
    new_tensor->num_dims = num_dims;
    for(uint8_t dim_index = 0; dim_index < num_dims; dim_index++){
        new_tensor->dims[dim_index] = dims[dim_index];
    }
    return new_tensor;
}

// TODO - ensure this is inlined
tensor_t* _new_tensor_like(const tensor_t* old_tensor){
    return _new_tensor(old_tensor->num_dims, old_tensor->dims);
}

// the same as _new_tensor_like (for now)
tensor_t* _new_tensor_zeros_like(const tensor_t* old_tensor){
    return _new_tensor(old_tensor->num_dims, old_tensor->dims);
}


tensor_t* _copy_tensor(const tensor_t* old_tensor){
    tensor_t* new_tensor = _new_tensor_like(old_tensor);
    memcpy(new_tensor->data, old_tensor->data, _tensor_get_size_in_bytes(old_tensor));
    return new_tensor;
}

/**
 * PROPERTIES
*/

// returns True iff the tensors dimensions are 1
bool _tensor_is_scalar(tensor_t* tensor){
    return (tensor->num_dims == 1) && (tensor->dims[0] == 1);
}

/**
 * NON-INLINED SETTERS/MUTATORS
*/

void _tensor_set_to_scalar_value(tensor_t* tensor, tensor_entry_t value){
    tensor_size_t tensor_size = _tensor_get_size(tensor);
    for(tensor_size_t index = 0; index < tensor_size; index++){
        _tensor_set_entry(tensor, index, value);
    }
}

void _tensor_multiply_by_scalar_value(tensor_t* tensor, tensor_entry_t value){
    tensor_size_t tensor_size = _tensor_get_size(tensor);
    for(tensor_size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = value * _tensor_get_entry(tensor, index);
        _tensor_set_entry(tensor, index, new_value);
    }
}

void _tensor_multiply_existing(tensor_t* multiplicand, tensor_t* multiplier){
    tensor_size_t tensor_size = _tensor_get_size(multiplicand);
    for(tensor_size_t index = 0; index < tensor_size; index++){
        tensor_entry_t new_value = _tensor_get_entry(multiplicand, index) * _tensor_get_entry(multiplier, index);
        _tensor_set_entry(multiplicand, index, new_value);
    }
}

void _tensor_in_place_apply_index_fn(tensor_t* tensor, tensor_index_fn_t index_fn){
    tensor_size_t tensor_size = _tensor_get_size(tensor);
    for(tensor_size_t index = 0; index < tensor_size; index++){
        tensor_entry_t entry_value = (*index_fn)(index);
        _tensor_set_entry(tensor, index, entry_value);
    }
}

void _tensor_in_place_apply_entry_fn(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn){
    tensor_size_t tensor_size = _tensor_get_size(tensor);
    for(tensor_size_t index = 0; index < tensor_size; index++){
        tensor_entry_t entry_old_value = _tensor_get_entry(tensor, index);
        tensor_entry_t entry_new_value = (*entry_fn)(entry_old_value);
        _tensor_set_entry(tensor, index, entry_new_value);
    }
}

static inline tensor_t* _tensor_apply_index_fn(tensor_t* tensor, tensor_entry_unary_fn_t index_fn){
    tensor_t* new_tensor = _copy_tensor(tensor);
    _tensor_in_place_apply_index_fn(new_tensor, index_fn);
    return new_tensor;
}

static inline tensor_t* _tensor_apply_entry_fn(tensor_t* tensor, tensor_entry_unary_fn_t entry_fn){
    tensor_t* new_tensor = _copy_tensor(tensor);
    _tensor_in_place_apply_entry_fn(new_tensor, entry_fn);
    return new_tensor;
}




/**
 * PRINTING
*/

void _display_tensor_dim_1(tensor_t* tensor){
    printf("Dimensions: %llu\n", tensor->dims[0]);
    for(tensor_size_t dim0_index = 0; dim0_index < tensor->dims[0]; dim0_index++){
        printf("%f ", tensor->data[dim0_index]);
    }
    printf("\n");
}

void _display_tensor_dim_2(tensor_t* tensor){
    printf("Dimensions: %llu x %llu\n", tensor->dims[0], tensor->dims[1]);
    for(tensor_size_t dim0_index = 0; dim0_index < tensor->dims[0]; dim0_index++){
        for(tensor_size_t dim1_index = 0; dim1_index < tensor->dims[1]; dim1_index++){
            tensor_size_t index = dim0_index * tensor->dims[1] + dim1_index;
            printf("%f ", tensor->data[index]);
        }
        printf("\n");
    } 
}

void _display_tensor_dim_3(tensor_t* tensor){
    printf("Dimensions: %llu x %llu x %llu\n", tensor->dims[0], tensor->dims[1], tensor->dims[2]);
    for(tensor_size_t dim0_index = 0; dim0_index < tensor->dims[0]; dim0_index++){
        for(tensor_size_t dim1_index = 0; dim1_index < tensor->dims[1]; dim1_index++){
            for(tensor_size_t dim2_index = 0; dim2_index < tensor->dims[2]; dim2_index++){
                tensor_size_t index = dim0_index * tensor->dims[1] * tensor->dims[2] + dim1_index * tensor->dims[2] + dim2_index;
                printf("%f ", tensor->data[index]);
            }
            printf("\n");
        }
        printf("\n");
    } 
}

void _display_tensor(tensor_t* tensor){
    switch(tensor->num_dims){
        case 1:
            _display_tensor_dim_1(tensor);
            break;
        case 2:
            _display_tensor_dim_2(tensor);
            break;
        case 3:
            _display_tensor_dim_3(tensor);
            break;
    }
}

/**
 * VIEWS - TODO
*/

// returns true if and only if tensor dimensions match exactly
// used as a pre-check for component-wise operations
bool _tensor_exact_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
    if(left_tensor->num_dims != left_tensor->num_dims){
        return 0;
    }
    for(int dim_index = 0; dim_index < left_tensor->num_dims; dim_index++){
        if(left_tensor->dims[dim_index] != right_tensor->dims[dim_index]){
            return 0;
        }
    }
    return 1;
}

// TODO: returns true iff tensors can be broadcasted in the same manner as in numpy/PyTorch
bool _tensor_broadcast_componentwise_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
    if(left_tensor->num_dims < right_tensor->num_dims){
        return _tensor_broadcast_componentwise_compatible(right_tensor, left_tensor);
    }
    // left_tensor->num_dims >= right_tensor->num_dims
    uint8_t dim_offset = left_tensor->num_dims - right_tensor->num_dims;
    for(uint8_t dim_index = 0; dim_index < right_tensor->num_dims; dim_index++){
        if(left_tensor->dims[dim_offset + dim_index] != right_tensor->dims[dim_index]){
            return 0;
        }
    }
    return 1;
}

/**
 * tensor_entry_t x tensor_entry_t -> tensor_entry_t
 * tensor_entry_t -> tensor_entry_t
*/

static inline tensor_entry_t _tensor_entry_add(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry + right_entry;
}

static inline tensor_entry_t _tensor_entry_multiply(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry * right_entry;
}

static inline tensor_entry_t _tensor_entry_subtract(tensor_entry_t left_entry, tensor_entry_t right_entry) {
    return left_entry - right_entry;
}

static inline tensor_entry_t _tensor_entry_abs_grad(tensor_entry_t entry){
    return entry >= 0 ? 1 : -1;
}

static inline tensor_entry_t _tensor_entry_abs(tensor_entry_t entry){
    return entry >= 0 ? entry : -entry;
}

void _tensor_in_place_broadcast_fn(tensor_t* dest_tensor, const tensor_t* source_tensor1, const tensor_t* source_tensor2, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    ASSERT(_tensor_broadcast_componentwise_compatible(source_tensor1, source_tensor2), "Tensors are not broadcast compatible!\n");
    if(source_tensor1->num_dims < source_tensor2->num_dims){
        return _tensor_in_place_broadcast_fn(dest_tensor, source_tensor2, source_tensor1, tensor_entry_binary_fn);
    }
    tensor_t* large_tensor = source_tensor1;
    tensor_t* small_tensor = source_tensor2;
    uint8_t dim_offset = large_tensor->num_dims - small_tensor->num_dims;
    tensor_size_t num_small = 1;
    for(uint8_t dim_index = 0; dim_index < dim_offset; dim_index++){
        num_small *= large_tensor->dims[dim_index];
    }
    tensor_size_t small_tensor_size = _tensor_get_size(small_tensor);
    for(tensor_size_t small_outer_index = 0; small_outer_index < num_small; small_outer_index++){
        tensor_size_t _large_index = small_outer_index * small_tensor_size;
        for(tensor_size_t small_inner_index = 0; small_inner_index < small_tensor_size; small_inner_index++){
            tensor_size_t large_index = _large_index + small_inner_index;
            tensor_entry_t large_entry = _tensor_get_entry(large_tensor, large_index);
            tensor_entry_t small_entry = _tensor_get_entry(small_tensor, small_inner_index);
            tensor_entry_t new_entry = (*tensor_entry_binary_fn)(large_entry, small_entry);
            _tensor_set_entry(dest_tensor, large_index, new_entry);
        }
    }
}

// TODO : allow for broadcasting of different sizes
tensor_t* _tensor_broadcast_fn(tensor_t* left_tensor, tensor_t* right_tensor, tensor_entry_binary_fn_t tensor_entry_binary_fn){
    // ensure that left_tensor->num_dims >= right_tensor->num_dims
    tensor_t* new_tensor = (left_tensor->num_dims > right_tensor->num_dims) ? _new_tensor_like(left_tensor) : _new_tensor_like(right_tensor);
    _tensor_in_place_broadcast_fn(new_tensor, left_tensor, right_tensor, tensor_entry_binary_fn);
    return new_tensor;
}

/**
 * MUTATING FUNCTIONS
*/

void _tensor_add_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor){
    _tensor_in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &_tensor_entry_add);
}

void _tensor_subtract_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor){
    _tensor_in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &_tensor_entry_subtract);
}

void _tensor_multiply_to_existing(tensor_t* left_tensor, const tensor_t* right_tensor){
    _tensor_in_place_broadcast_fn(left_tensor, left_tensor, right_tensor, &_tensor_entry_multiply);
}

/**
 *  NON-MUTATING FUNCTIONS
*/

// return new tensor which is the result of component-wise addition 
// of left_tensor and right_tensor
// assumes that left_tensor and right_tensor are compatible
tensor_t* _tensor_add(const tensor_t* left_tensor, const tensor_t* right_tensor){
    return _tensor_broadcast_fn(left_tensor, right_tensor, &_tensor_entry_add);
}

tensor_t* _tensor_subtract(const tensor_t* left_tensor, const tensor_t* right_tensor){
    return _tensor_broadcast_fn(left_tensor, right_tensor, &_tensor_entry_subtract);
}

tensor_t* _tensor_multiply(const tensor_t* left_tensor, const tensor_t* right_tensor){
    return _tensor_broadcast_fn(left_tensor, right_tensor, &_tensor_entry_multiply);
}

tensor_t* _tensor_abs_grad(const tensor_t* tensor){
    tensor_t* new_tensor = _copy_tensor(tensor);
    _tensor_in_place_apply_entry_fn(new_tensor, &_tensor_entry_abs_grad);
    return new_tensor;
}

tensor_t* _tensor_abs(const tensor_t* tensor){
    tensor_t* new_tensor = _copy_tensor(tensor);
    _tensor_in_place_apply_entry_fn(new_tensor, &_tensor_entry_abs);
    return new_tensor;
}
