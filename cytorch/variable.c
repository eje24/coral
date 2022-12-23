#include "variable.h"
#include "tensor.h"
#include <stdio.h>

variable_t _new_variable(tensor_t tensor){
    return (variable_t) {.tensor = tensor, .gradient = _new_tensor_zeros_like(tensor)};
}

// creates a new variable with tensor of the same dimensions as old_variable
variable_t _new_variable_like(variable_t old_variable){
    tensor_t new_tensor = _new_tensor_like(old_variable.tensor);
    return _new_variable(new_tensor);
}

// creates a new variable by copying the contents of old_variable
variable_t _copy_variable(variable_t old_variable){
    tensor_t new_tensor = _copy_tensor(old_variable.tensor);
    return _new_variable(new_tensor);
}

// performs component-wise addition
variable_t add(variable_t left_variable, variable_t right_variable){
    tensor_t new_tensor = _tensor_add(left_variable.tensor, right_variable.tensor);
    return _new_variable(new_tensor);
}