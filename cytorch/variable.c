#include "variable.h"
#include "tensor.h"

variable_t _new_variable(tensor_t tensor){
    return (variable_t) {.tensor = tensor, .gradient = 0};
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