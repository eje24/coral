#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <stdbool.h>

// wrapper around tensor
// intended to store metadata necessary for backpropagation
typedef struct {
    tensor_t tensor;
    tensor_t gradient; 
} variable_t;

variable_t _new_variable_from_tensor(tensor_t tensor);
variable_t new_variable(tensor_size_t num_rows, tensor_size_t num_columns);
variable_t new_variable_like(variable_t old_variable);
variable_t copy_variable(variable_t old_variable);

static inline tensor_entry_t get_entry(variable_t variable, tensor_size_t index){
    return _tensor_get_entry(variable.tensor, index);
}

static inline tensor_entry_t get_entry_row_column(variable_t variable, tensor_size_t row, tensor_size_t column){
    return _tensor_get_entry_row_column(variable.tensor, row, column);
}

static inline void set_entry(variable_t variable, tensor_size_t index){
    _tensor_set_entry(variable.tensor, index);
}

static inline void set_entry_row_column(variable_t variable, tensor_size_t row, tensor_size_t column){
    _tensor_set_entry_row_column(variable.tensor, row, column);
}

#endif // VARIABLE_H