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

void _unary_set_grad_meta(variable_t* child, variable_t* parent, variable_grad_op_t grad_op){
    diff_arg_t* diff_arg = _new_diff_arg(parent, grad_op);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->num_args = 1;
    new_grad_meta->args[0] = diff_arg;
}

void _binary_set_grad_meta(variable_t* child, variable_t* parent1, variable_t* parent2, variable_binary_grad_op_t grad_op1, variable_binary_grad_op_t grad_op2){
    diff_arg_t* diff_arg1 = _new_diff_arg(parent1, grad_op1);
    diff_arg_t* diff_arg2 = _new_diff_arg(parent2, grad_op2);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->num_args = 2;
    new_grad_meta->args[0] = diff_arg1;
    new_grad_meta->args[1] = diff_arg2;
    child->grad_meta = new_grad_meta;
}

variable_t* add(variable_t* left_variable, variable_t* right_variable){
    return _add(left_variable, right_variable, true);
}

variable_t* subtract(variable_t* left_variable, variable_t* right_variable){
    return _subtract(left_variable, right_variable, true);
}

variable_t* multiply(variable_t* left_variable, variable_t* right_variable){
    return _multiply(left_variable, right_variable, true);
}

/**
 * GRADIENTS: return grad with respect to arg, possible as a function of both arg and other_arg
*/
tensor_t* _add_grad(variable_t* arg, variable_t* other_arg, variable_t* child){
    return _copy_tensor(child->gradient);
}

tensor_t* _subtract_grad(variable_t* arg, variable_t* other_arg, variable_t* child){
    tensor_t* child_grad = _copy_tensor(child->gradient);
    _tensor_multiply_by_scalar_value(child_grad, -1);
    return child_grad;
}

tensor_t* _multiply_grad(variable_t* arg, variable_t* other_arg, variable_t* child){
    tensor_t* arg_grad = _copy_tensor(child->gradient);
    _tensor_multiply_existing(arg_grad, other_arg->tensor);
    return arg_grad;
}

// performs component-wise addition
variable_t* _add(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad){
    tensor_t* new_tensor = _tensor_add(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_add_grad, &_add_grad);
    } 
    return new_variable;
}

// performs component-wise addition
variable_t* _subtract(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad){
    tensor_t* new_tensor = _tensor_subtract(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_subtract_grad, &_subtract_grad);
    } 
    return new_variable;
}

// performs component-wise addition
variable_t* _multiply(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad){
    tensor_t* new_tensor = _tensor_multiply(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_multiply_grad, &_multiply_grad);
    } 
    return new_variable;
}

