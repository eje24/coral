#include "variable.h"
#include "tensor.h"
#include <stdio.h>

/**
 * CONSTRUCTORS
*/

variable_t _new_variable_from_tensor(tensor_t tensor){
    return (variable_t) {.tensor = tensor, .gradient = _new_tensor_zeros_like(tensor)};
}

variable_t new_variable(tensor_size_t num_rows, tensor_size_t num_columns){
    tensor_t new_tensor = _new_tensor(num_rows, num_columns);
    return _new_variable_from_tensor(new_tensor);
}

// creates a new variable with tensor of the same dimensions as old_variable
variable_t new_variable_like(variable_t old_variable){
    tensor_t new_tensor = _new_tensor_like(old_variable.tensor);
    return _new_variable_from_tensor(new_tensor);
}

// creates a new variable by copying the contents of old_variable
variable_t copy_variable(variable_t old_variable){
    tensor_t new_tensor = _copy_tensor(old_variable.tensor);
    return _new_variable_from_tensor(new_tensor);
}

/**
 * FUNCTIONS
*/

// performs component-wise addition
variable_t add(variable_t left_variable, variable_t right_variable){
    tensor_t new_tensor = _tensor_add(left_variable.tensor, right_variable.tensor);
    return _new_variable_from_tensor(new_tensor);
}

// performs component-wise addition
variable_t subtract(variable_t left_variable, variable_t right_variable){
    tensor_t new_tensor = _tensor_subtract(left_variable.tensor, right_variable.tensor);
    return _new_variable_from_tensor(new_tensor);
}

// performs component-wise addition
variable_t multiply(variable_t left_variable, variable_t right_variable){
    tensor_t new_tensor = _tensor_multiply(left_variable.tensor, right_variable.tensor);
    return _new_variable_from_tensor(new_tensor);
}

