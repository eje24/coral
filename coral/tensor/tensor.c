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

static inline tensor_size_t _tensor_get_size(tensor_t* tensor){
    return _get_size_from_dims(tensor->num_dims, tensor->dims);
}

static inline size_t _tensor_get_size_in_bytes(tensor_t* tensor){
    return _tensor_get_size(tensor) * sizeof(tensor_entry_t);
}

// create new tensor
// entries are set to zero by default
tensor_t* _new_tensor(uint8_t num_dims, tensor_size_t* dims){
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
tensor_t* _new_tensor_like(tensor_t* old_tensor){
    return _new_tensor(old_tensor->num_dims, old_tensor->dims);
}

// the same as _new_tensor_like (for now)
tensor_t* _new_tensor_zeros_like(tensor_t* old_tensor){
    return _new_tensor(old_tensor->num_dims, old_tensor->dims);
}


tensor_t* _copy_tensor(tensor_t* old_tensor){
    tensor_t* new_tensor = _new_tensor_like(old_tensor);
    memcpy(new_tensor->data, old_tensor->data, _tensor_get_size_in_bytes(old_tensor));
    return new_tensor;
}

// void _populate_tensor(tensor_t* tensor, tensor_row_column_fn_t row_column_fn){
//     for(tensor_size_t row = 0; row < tensor->num_rows; row++){
//         for(tensor_size_t column = 0; column < tensor->num_columns; column++){
//             tensor_entry_t new_entry = (*row_column_fn)(row, column);
//             _tensor_set_entry_row_column(tensor, row, column, new_entry);
//         }
//     }
// }

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

/**
 *  FUNCTIONS
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

// TODO
// bool _tensor_broadcast_matmul_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
//     return 1;
// }

// returns true if and only if internal dimensions match
// so that left_tensor @ right_tensor is a valid matrix multiplication
// static inline bool _tensor_internal_compatible(tensor_t* left_tensor, tensor_t* right_tensor){
//     return (left_tensor->num_columns == right_tensor->num_rows);
// }

static inline tensor_entry_t _scalar_add(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar + right_scalar;
}

static inline tensor_entry_t _scalar_multiply(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar * right_scalar;
}

static inline tensor_entry_t _scalar_subtract(tensor_entry_t left_scalar, tensor_entry_t right_scalar) {
    return left_scalar - right_scalar;
}

// TODO : allow for broadcasting of different sizes
static inline tensor_t* _tensor_broadcast_scalar_fn(tensor_t* left_tensor, tensor_t* right_tensor, tensor_entry_fn_t tensor_scalar_fn){
    DEBUG_ASSERT(_tensor_broadcast_compatible(left_tensor, right_tensor), "Tensors are not broadcast compatible!\n");
    // ensure that left_tensor->num_dims >= right_tensor->num_dims
    if(left_tensor->num_dims < right_tensor->num_dims){
        return _tensor_broadcast_scalar_fn(right_tensor, left_tensor, tensor_scalar_fn);
    }
    tensor_t* large_tensor = left_tensor;
    tensor_t* small_tensor = right_tensor;
    tensor_t* new_tensor = _new_tensor_like(large_tensor);
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
            tensor_entry_t new_entry = (*tensor_scalar_fn)(large_entry, small_entry);
            _tensor_set_entry(new_tensor, large_index, new_entry);
        }
    }
    return new_tensor;
}

// return new tensor which is the result of component-wise addition 
// of left_tensor and right_tensor
// assumes that left_tensor and right_tensor are compatible
tensor_t* _tensor_add(tensor_t* left_tensor, tensor_t* right_tensor){
    return _tensor_broadcast_scalar_fn(left_tensor, right_tensor, &_scalar_add);
}

tensor_t* _tensor_subtract(tensor_t* left_tensor, tensor_t* right_tensor){
    return _tensor_broadcast_scalar_fn(left_tensor, right_tensor, &_scalar_subtract);
}

tensor_t* _tensor_multiply(tensor_t* left_tensor, tensor_t* right_tensor){
    return _tensor_broadcast_scalar_fn(left_tensor, right_tensor, &_scalar_multiply);
}
