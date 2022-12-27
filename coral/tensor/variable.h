#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <stdbool.h>

// wrapper around tensor
// intended to store metadata necessary for backpropagation
typedef struct {
    tensor_t* tensor;
    tensor_t* gradient;
} variable_t;

variable_t* _new_variable_from_tensor(tensor_t* tensor);
variable_t* new_variable(int num_dims, ...);
variable_t* new_variable_like(variable_t* old_variable);
variable_t* copy_variable(variable_t* old_variable);

void set_to_scalar(variable_t* variable, tensor_entry_t value);

variable_t* add(variable_t* left_variable, variable_t* right_variable);
variable_t* subtract(variable_t* left_variable, variable_t* right_variable);
variable_t* multiply(variable_t* left_variable, variable_t* right_variable);

void display_variable(variable_t* variable);
void display_variable_with_gradient(variable_t* variable);

static inline tensor_entry_t get_entry(variable_t* variable, tensor_size_t index){
    return _tensor_get_entry(variable->tensor, index);
}

static inline void set_entry(variable_t* variable, tensor_size_t index, tensor_entry_t value){
    _tensor_set_entry(variable->tensor, index, value);
}

static inline void set_to_scalar_value(variable_t* variable, tensor_entry_t value){
    _tensor_set_to_scalar_value(variable->tensor, value);
}

#endif // VARIABLE_H