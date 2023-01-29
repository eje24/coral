#include "variable.h"
#include "tensor.h"
#include "grad.h"
#include "shape.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/**
 * CONSTRUCTORS
*/

variable_t* variable_new_from_tensor(tensor_t* tensor){
    variable_t* new_variable = (variable_t *) malloc(sizeof(variable_t));
    new_variable->tensor = tensor;
    new_variable->gradient = tensor_new_zeros_like(tensor);
    new_variable->grad_meta = grad_meta_new();
    return new_variable;
}

variable_t* variable_new(int num_dims, ...){
    // parse dim arguments
    size_t dims[num_dims];
    va_list dim_args;
    va_start(dim_args, num_dims);
    for(uint8_t dim_index = 0; dim_index < num_dims; dim_index++){
        dims[dim_index] = va_arg(dim_args, size_t);
    }
    va_end(dim_args);
    shape_t* shape = shape_new(num_dims, &dims[0]);
    tensor_t* new_tensor = tensor_new(shape);
    return variable_new_from_tensor(new_tensor);
}

void variable_in_place_view_as(variable_t* variable, int num_dims, ...){
    // parse dim arguments
    size_t dims[num_dims];
    va_list dim_args;
    va_start(dim_args, num_dims);
    for(uint8_t dim_index = 0; dim_index < num_dims; dim_index++){
        dims[dim_index] = va_arg(dim_args, size_t);
    }
    va_end(dim_args);
    shape_t* shape = shape_new(num_dims, &dims[0]);
    tensor_in_place_view_as_shape(variable->tensor, shape);
}

void variable_in_place_view_as_shape(variable_t* variable, shape_t* new_shape){
    tensor_in_place_view_as_shape(variable->tensor, new_shape);
}


variable_t* variable_view_as(variable_t* variable, int num_dims, ...){
    // parse dim arguments
    size_t dims[num_dims];
    va_list dim_args;
    va_start(dim_args, num_dims);
    for(uint8_t dim_index = 0; dim_index < num_dims; dim_index++){
        dims[dim_index] = va_arg(dim_args, size_t);
    }
    va_end(dim_args);
    shape_t* shape = shape_new(num_dims, &dims[0]);
    tensor_t* new_tensor = tensor_view_as_shape(variable->tensor, shape);
    return variable_new_from_tensor(new_tensor);
}

variable_t* variable_view_as_shape(variable_t* variable, shape_t* new_shape){
    return variable_new_from_tensor(tensor_view_as_shape(variable->tensor, new_shape));
}


// creates a new variable with tensor of the same dimensions as old_variable
variable_t* variable_new_like(variable_t* old_variable){
    tensor_t* new_tensor = tensor_new_like(old_variable->tensor);
    return variable_new_from_tensor(new_tensor);
}

// creates a new variable with tensor of the same dimensions as old_variable
variable_t* variable_new_like_with_value(variable_t* old_variable, tensor_entry_t value){
    tensor_t* new_tensor = tensor_new_like_with_value(old_variable->tensor, value);
    return variable_new_from_tensor(new_tensor);
}

// creates a new variable by copying the contents of old_variable
variable_t* variable_copy(variable_t* old_variable){
    tensor_t* new_tensor = tensor_copy(old_variable->tensor);
    return variable_new_from_tensor(new_tensor);
}

/**
 * COMPARATORS
*/

// returns True if and only if the (tensor_t) contents are the same
// irrespective of autograd information
bool variable_equal(variable_t* left_variable, variable_t* right_variable){
    return tensor_equal(left_variable->tensor, right_variable->tensor);
}

// returns True if and only if the variables share the same data
// irrespective of shape 
bool variable_alias(variable_t* left_variable, variable_t* right_variable){
    return (left_variable->tensor->data == right_variable->tensor->data);
}

/**
 * MUTATE EXISTING VARIABLE
*/

void variable_in_place_apply_index_fn(variable_t* variable, tensor_index_fn_t index_fn){
    tensor_in_place_apply_index_fn(variable->tensor, index_fn);
}

/**
 * PRINTING
*/

void variable_display(variable_t* variable, char* name){
    printf("Name: %s\n", name);
    printf("Tensor:\n");
    tensor_display(variable->tensor);
}

void variable_display_with_gradient(variable_t* variable, char* name){
    printf("Name: %s\n", name);
    printf("Tensor:\n");
    tensor_display(variable->tensor);
    printf("Gradient:\n");
    tensor_display(variable->gradient);
}

void variable_set_to_scalar(variable_t* variable, tensor_entry_t value){
    tensor_set_to_scalar_value(variable->tensor, value);
}

/**
 * GRADIENTS: return grad with respect to input, possible as a function of both input and other_input
*/


/**
 * INTERNAL ATOMIC FUNCTIONS
 * 
 * these functios explicitly update the gradient graph
 * other functios which are compositions of these atomic functions
 * rely on these functions to update the computation graph
*/

static inline tensor_t* add_backwards_grad(variable_t* input, variable_t* other_input, variable_t* output){
    UNUSED(input);
    UNUSED(other_input);
    return tensor_copy(output->gradient);
}

// performs component-wise addition
variable_t* add(variable_t* left_variable, variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = tensor_add(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = variable_new_from_tensor(new_tensor);
    if(use_grad){
        set_binary_grad_meta(new_variable, left_variable, right_variable, &add_backwards_grad, &add_backwards_grad);
    } 
    return new_variable;
}

tensor_t* subtract_backwards_grad(variable_t* input, variable_t* other_input, variable_t* output){
    UNUSED(input);
    UNUSED(other_input);
    tensor_t* output_grad = tensor_copy(output->gradient);
    tensor_in_place_multiply_by_scalar(output_grad, -1);
    return output_grad;
}

// performs component-wise addition
variable_t* subtract(variable_t* left_variable, variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = tensor_subtract(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = variable_new_from_tensor(new_tensor);
    if(use_grad){
        set_binary_grad_meta(new_variable, left_variable, right_variable, &subtract_backwards_grad, &subtract_backwards_grad);
    } 
    return new_variable;
}

tensor_t* multiply_backwards_grad(variable_t* input, variable_t* other_input, variable_t* output){
    UNUSED(input);
    return tensor_multiply(output->gradient, other_input->tensor);
}

// returns a new variable whose value is given by the sum of left_variable and right_variable
variable_t* multiply(variable_t* left_variable, variable_t* right_variable, bool use_grad){
    tensor_t* new_tensor = tensor_multiply(left_variable->tensor, right_variable->tensor);
    variable_t* new_variable = variable_new_from_tensor(new_tensor);
    if(use_grad){
        set_binary_grad_meta(new_variable, left_variable, right_variable, &multiply_backwards_grad, &multiply_backwards_grad);
    } 
    return new_variable;
}

tensor_t* square_backwards_grad(variable_t* variable, variable_t* result){
    return tensor_multiply(tensor_multiply_by_scalar(variable->tensor, 2.0), result->gradient);
}

// note that square is equivalent (in terms of correctness of result and grad meta update) to multiply

variable_t* square(variable_t* variable, bool use_grad){
    variable_t* new_variable = variable_new_from_tensor(tensor_multiply(variable->tensor, variable->tensor));
    if(use_grad){
        set_unary_grad_meta(new_variable, variable, &square_backwards_grad);
    }
    return new_variable;
}

tensor_t* abs_value_backwards_grad(variable_t* input, variable_t* result){
    return tensor_multiply(tensor_abs(input->tensor), result->gradient);
}

// returns a new variable whose value is given by the absolute value of variable
static variable_t* abs_value(variable_t* variable, bool use_grad){
    tensor_t* new_tensor = tensor_abs(variable->tensor);
    variable_t* new_variable =  variable_new_from_tensor(new_tensor);
    if(use_grad){
        set_unary_grad_meta(new_variable, variable, &abs_value_backwards_grad);
    }
    return new_variable;
}

tensor_t* sum_backwards_grad(variable_t* input, variable_t* result){
    return tensor_multiply(tensor_sum_grad(input->tensor), result->gradient);

}

variable_t* sum(variable_t* variable, bool use_grad){
    variable_t* new_variable = variable_new_from_tensor(tensor_sum(variable->tensor));
    if(use_grad){
        set_unary_grad_meta(new_variable, variable, &sum_backwards_grad);
    }
    return new_variable;
}

tensor_t* mean_backwards_grad(variable_t* input, variable_t* result){
    return tensor_multiply(tensor_mean_grad(input->tensor), result->gradient);
}

variable_t* mean(variable_t* variable, bool use_grad){
    variable_t* new_variable = variable_new_from_tensor(tensor_mean(variable->tensor));
    if(use_grad){
        set_unary_grad_meta(new_variable, variable, &mean_backwards_grad);
    }
    return new_variable;
}

/**
 * EXTERNAL FUNCTIONS
*/

variable_t* variable_add(variable_t* left_variable, variable_t* right_variable){
    return add(left_variable, right_variable, true);
}

variable_t* variable_subtract(variable_t* left_variable, variable_t* right_variable){
    return subtract(left_variable, right_variable, true);
}

variable_t* variable_multiply(variable_t* left_variable, variable_t* right_variable){
    return multiply(left_variable, right_variable, true);
}

variable_t* variable_square(variable_t* variable){
    return square(variable, true);
}

variable_t* variable_abs_value(variable_t* variable){
    return abs_value(variable, true);
}

variable_t* variable_sum(variable_t* variable){
    return sum(variable, true);
}

variable_t* variable_mean(variable_t* variable){
    return mean(variable, true);
}

/**
 * LOSS FUNCTIONS
*/

// mean absolute error
variable_t* variable_mae_loss(variable_t* actual, variable_t* expected){
    return variable_mean(variable_abs_value(variable_subtract(actual, expected)));
}

// mean squared error
variable_t* variable_mse_loss(variable_t* actual, variable_t* expected){
    return variable_mean(variable_square(variable_subtract(actual, expected)));
}


