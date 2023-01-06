#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <stdbool.h>

// wrapper around tensor
// intended to store metadata necessary for backpropagation
typedef struct variable variable_t;
typedef struct grad_meta grad_meta_t;

struct variable {
    tensor_t* tensor;
    tensor_t* gradient;
    grad_meta_t* grad_meta;
};

variable_t* variable_new(int num_dims, ...);
variable_t* variable_new_from_tensor(tensor_t* tensor);
variable_t* variable_new_like(variable_t* old_variable);
variable_t* variable_copy(variable_t* old_variable);

static inline bool is_scalar(variable_t* variable){
    return tensor_is_scalar(variable->tensor);
}

typedef variable_t* (* variable_binary_op_t)(variable_t* left_variable, variable_t* right_variable);
typedef variable_t* (* variable_unary_op_t)(variable_t* left_variable, variable_t* right_variable);
typedef tensor_t* (* variable_binary_grad_op_t)(variable_t* arg, variable_t* other_arg, variable_t* result);
typedef tensor_t* (* variable_unary_grad_op_t)(variable_t* arg, variable_t* result);
typedef void (* generic_op_t)();

#define variable_grad_op_t generic_op_t

// differentiable argument
typedef struct {
    variable_t* arg;
    variable_grad_op_t grad_op;
} diff_arg_t;

static inline diff_arg_t* diff_arg_new(variable_t* arg, variable_grad_op_t grad_op){
    diff_arg_t* new_diff_arg = (diff_arg_t*) malloc(sizeof(diff_arg_t));
    new_diff_arg->arg = arg;
    new_diff_arg->grad_op = grad_op;
    return new_diff_arg;
}

struct grad_meta{
    int ref_count;
    int num_args; // 0 for leaf
    diff_arg_t* args[2];
};

static inline grad_meta_t* _default_grad_meta(){
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_args = 0;
    return new_grad_meta;
}

variable_t* variable_add(variable_t* left_variable, variable_t* right_variable);
variable_t* variable_subtract(variable_t* left_variable, variable_t* right_variable);
variable_t* variable_multiply(variable_t* left_variable, variable_t* right_variable);
variable_t* variable_abs_value(variable_t* variable);
variable_t* variable_sum(variable_t* variable);

void variable_display(variable_t* variable);
void variable_display_with_gradient(variable_t* variable);

static inline tensor_entry_t get_entry(variable_t* variable, size_t index){
    return tensor_get_entry(variable->tensor, index);
}

static inline void set_entry(variable_t* variable, size_t index, tensor_entry_t value){
    tensor_set_entry(variable->tensor, index, value);
}

static inline void variable_set_to_scalar_value(variable_t* variable, tensor_entry_t value){
    tensor_set_to_scalar_value(variable->tensor, value);
}

#endif // VARIABLE_H