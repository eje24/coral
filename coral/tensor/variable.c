#include "variable.h"
#include "tensor.h"
#include "grad.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/**
 * CONSTRUCTORS
*/

variable_t* _new_variable_from_tensor(const tensor_t* tensor){
    variable_t* new_variable = (variable_t *) malloc(sizeof(variable_t));
    new_variable->tensor = tensor;
    new_variable->gradient = _new_tensor_zeros_like(tensor);
    new_variable->grad_meta = _default_grad_meta();
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
variable_t* new_variable_like(const variable_t* old_variable){
    tensor_t* new_tensor = _new_tensor_like(old_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

// creates a new variable by copying the contents of old_variable
variable_t* copy_variable(const variable_t* old_variable){
    tensor_t* new_tensor = _copy_tensor(old_variable->tensor);
    return _new_variable_from_tensor(new_tensor);
}

/**
 * PRINTING
*/

void display_variable(const variable_t* variable){
    printf("Tensor:\n");
    _display_tensor(variable->tensor);
}

void display_variable_with_gradient(const variable_t* variable){
    printf("Tensor:\n");
    _display_tensor(variable->tensor);
    printf("Gradient:\n");
    _display_tensor(variable->gradient);
}

void set_to_scalar(variable_t* variable, tensor_entry_t value){
    _tensor_set_to_scalar_value(variable->tensor, value);
}

/**
 * EXTERNAL FUNCTIONS
*/

variable_t* add(const variable_t* left_variable, const variable_t* right_variable){
    return _add(left_variable, right_variable, true);
}

variable_t* subtract(const variable_t* left_variable, const variable_t* right_variable){
    return _subtract(left_variable, right_variable, true);
}

variable_t* multiply(const variable_t* left_variable, const variable_t* right_variable){
    return _multiply(left_variable, right_variable, true);
}

/**
 * GRADIENTS: return grad with respect to arg, possible as a function of both arg and other_arg
*/


/**
 * INTERNAL ATOMIC FUNCTIONS
 * 
 * these functios explicitly update the gradient graph
 * other functios which are compositions of these atomic functions
 * rely on these functions to update the computation graph
*/

static inline tensor_t* _add_grad(const variable_t* arg, const variable_t* other_arg, const variable_t* child){
    return _copy_tensor(child->gradient);
}

// performs component-wise addition
variable_t* _add(const variable_t* left_variable, const variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = _tensor_add(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_add_grad, &_add_grad);
    } 
    return new_variable;
}

tensor_t* _subtract_grad(const variable_t* arg, const variable_t* other_arg, const variable_t* child){
    tensor_t* child_grad = _copy_tensor(child->gradient);
    _tensor_multiply_by_scalar_value(child_grad, -1);
    return child_grad;
}

// performs component-wise addition
variable_t* _subtract(const variable_t* left_variable, const variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = _tensor_subtract(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_subtract_grad, &_subtract_grad);
    } 
    return new_variable;
}

tensor_t* _multiply_grad(const variable_t* arg, const variable_t* other_arg, const variable_t* child){
    tensor_t* arg_grad = _copy_tensor(child->gradient);
    _tensor_multiply_existing(arg_grad, other_arg->tensor);
    return arg_grad;
}

// returns a new variable whose value is given by the sum of left_variable and right_variable
variable_t* _multiply(const variable_t* left_variable, const variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = _tensor_multiply(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _binary_set_grad_meta(new_variable, left_variable, right_variable, &_multiply_grad, &_multiply_grad);
    } 
    return new_variable;
}

tensor_t* _abs_grad(const variable_t* arg, const variable_t* child){
    return _tensor_abs_grad(arg);
}

// returns a new variable whose value is given by the absolute value of variable
variable_t* _abs(const variable_t* variable, bool use_grad){
    tensor_t* new_tensor = _tensor_abs(variable->tensor);
    variable_t* new_variable =  _new_variable_from_tensor(new_tensor);
    if(use_grad){
        _unary_set_grad_meta(new_variable, variable, &_abs_grad);
    }
    return new_variable;
}

/**
 * LOSS FUNCTIONS
*/

variable_t* l1_loss();

variable_t* l2_loss();
