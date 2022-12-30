#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <stdbool.h>

// wrapper around tensor
// intended to store metadata necessary for backpropagation
typedef struct {
    tensor_t* tensor;
    tensor_t* gradient;
    grad_meta_t* grad_meta;
} variable_t;

variable_t* _new_variable_from_tensor(tensor_t* tensor);
variable_t* new_variable(int num_dims, ...);
variable_t* new_variable_like(variable_t* old_variable);
variable_t* copy_variable(variable_t* old_variable);

void set_to_scalar(variable_t* variable, tensor_entry_t value);

typedef variable_t* (* const variable_binary_op_t)(variable_t* left_entry, variable_t* right_entry);
typedef variable_t* (* const variable_unary_op_t)(variable_t* left_entry, variable_t* right_entry);
typedef tensor_t* (* const variable_binary_grad_op_t)(variable_t* left_entry, variable_t* right_entry, variable_t* child);
typedef tensor_t* (* const variable_unary_grad_op_t)(variable_t* entry, variable_t* child);
typedef void (* generic_op_t)();

#define variable_grad_op_t generic_op_t

// differentiable argument
typedef struct {
    variable_t* arg;
    variable_grad_op_t grad_op;
} diff_arg_t;

static inline diff_arg_t* _new_diff_arg(variable_t* arg, variable_grad_op_t grad_op){
    diff_arg_t* new_diff_arg = (diff_arg_t*) malloc(sizeof(diff_arg_t));
    new_diff_arg->arg = arg;
    new_diff_arg->grad_op = grad_op;
    return new_diff_arg;
}

typedef struct {
    int num_args; // 0 for leaf
    diff_arg_t* args[2];
} grad_meta_t;

variable_t* add(variable_t* left_variable, variable_t* right_variable);
variable_t* subtract(variable_t* left_variable, variable_t* right_variable);
variable_t* multiply(variable_t* left_variable, variable_t* right_variable);
variable_t* _add(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad);
variable_t* _subtract(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad);
variable_t* _multiply(variable_t* left_variable, variable_t* right_variable, uint8_t use_grad);

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