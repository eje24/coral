#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

static inline tensor_size_t get_size(tensor_t tensor){
    return tensor.num_rows * tensor.num_columns;
}

static inline size_t get_size_in_bytes(tensor_t tensor){
    return get_size(tensor) * sizeof(tensor_entry_t);
}

// create new tensor
tensor_t _new_tensor(tensor_size_t num_rows, tensor_size_t num_columns){
    uint32_t num_entries = num_rows * num_columns;
    tensor_entry_t* raw_data = (tensor_entry_t*) malloc(num_entries * sizeof(tensor_entry_t));
    return(tensor_t) {.data = raw_data, .num_rows = num_rows, .num_columns = num_columns};
}

tensor_t _new_tensor_like(tensor_t old_tensor){
    return _new_tensor(old_tenso.num_rows, old_tensor.num_columns);
}

tensor_t _copy_tensor(tensor_t old_tensor){
    tensor_t new_tensor = _new_tensor_like(old_tensor);
    memcpy(new_tensor->data, old_tensor->data, get_size_in_bytes(tensor));
    return new_tensor;
}

// return new tensor which is the result of component-wise addition 
// of left_tensor and right_tensor
// assumes that left_tensor and right_tensor are compatible
tensor_t _tensor_add(tensor_t left_tensor, tensor_t right_tensor){
    tensor_t new_tensor = _new_tensor_like(left_tensor)
}

// returns 0 if unsucessful, otherwise 1
inline bool set_tensor_entry(tensor_t* tensor, tensor_size_t row, tensor_size_t column, tensor_entry_t value){
    if(row >= tensor->num_rows || column >= tensor->num_columns){
        return 0;
    }
    tensor_size_t index = row * tensor->num_rows + column;
    tensor->data[index] = value;
    return 1;
}

