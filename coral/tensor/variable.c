#include "variable.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/**
 * CONSTRUCTORS
*/

variable_t* _new_variable_from_tensor(tensor_t* tensor){
    variable_t* new_variable = (variable_t *) malloc(sizeof(variable_t));
    new_variable->tensor = tensor;
    new_variable->gradient = _new_tensor_zeros_like(tensor);
    return new_variable;
}

variable_t* new_variable(int num_dims, ...){
    // parse dim arguments
    tensor_size_t dims[TENSOR_MAX_DIMS];
    va_list dim_args;
    va_start(dim_args, num_dims);
    for(uint8_t dim_index = 0; dim_index < num_dims; dim_index++){
        dims[dim_index] = va_arg(dim_args, tensor_size_t);
    }
    va_end(dim_args);
    tensor_t* new_tensor = _new_tensor(num_dims, &dims[0]);
    return _new_variable_from_tensor(new_tensor);
}

// creates a new variable with tensor of the same dimensions as old_variable
variable_t* new_variable_like(variable_t* old_variable){
    tensor_t* new_tensor = _new_tensor_like(old_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

// creates a new variable by copying the contents of old_variable
variable_t* copy_variable(variable_t* old_variable){
    tensor_t* new_tensor = _copy_tensor(old_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

/**
 * PRINTING
*/

void display_variable(variable_t* variable){
    printf("Tensor:\n");
    _display_tensor(variable->tensor);
}

void display_variable_with_gradient(variable_t* variable){
    printf("Tensor:\n");
    _display_tensor(variable->tensor);
    printf("Gradient:\n");
    _display_tensor(variable->gradient);
}

void set_to_scalar(variable_t* variable, tensor_entry_t value){
    _tensor_set_to_scalar_value(variable->tensor, value);
}

/**
 * FUNCTIONS
*/

// performs component-wise addition
variable_t* add(variable_t* left_variable, variable_t* right_variable){
    tensor_t* new_tensor = _tensor_add(left_variable->tensor, right_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

// performs component-wise addition
variable_t* subtract(variable_t* left_variable, variable_t* right_variable){
    tensor_t* new_tensor = _tensor_subtract(left_variable->tensor, right_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

// performs component-wise addition
variable_t* multiply(variable_t* left_variable, variable_t* right_variable){
    tensor_t* new_tensor = _tensor_multiply(left_variable->tensor, right_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

