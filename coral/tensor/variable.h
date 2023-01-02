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

variable_t* _new_variable_from_tensor(const tensor_t* tensor);
variable_t* new_variable(int num_dims, ...);
variable_t* new_variable_like(const variable_t* old_variable);
variable_t* copy_variable(const variable_t* old_variable);

static inline bool is_scalar(variable_t* variable){
    return _tensor_is_scalar(variable->tensor);
}

void set_to_scalar(variable_t* variable, tensor_entry_t value);

typedef variable_t* (* const variable_binary_op_t)(const variable_t* left_variable, const variable_t* right_variable);
typedef variable_t* (* const variable_unary_op_t)(const variable_t* left_variable, const variable_t* right_variable);
typedef tensor_t* (* const variable_binary_grad_op_t)(const variable_t* arg, const variable_t* other_arg, const variable_t* result);
typedef tensor_t* (* const variable_unary_grad_op_t)(const variable_t* arg, const variable_t* result);
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
    int ref_count;
    int num_args; // 0 for leaf
    diff_arg_t* args[2];
} grad_meta_t;

static inline grad_meta_t* _default_grad_meta(){
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_args = 0;
    return new_grad_meta;
}

variable_t* add(const variable_t* left_variable, const variable_t* right_variable);
variable_t* subtract(const variable_t* left_variable, const variable_t* right_variable);
variable_t* multiply(const variable_t* left_variable, const variable_t* right_variable);
variable_t* _add(const variable_t* left_variable, const variable_t* right_variable, bool use_grad);
variable_t* _subtract(const variable_t* left_variable, const variable_t* right_variable, bool use_grad);
variable_t* _multiply(const variable_t* left_variable, const variable_t* right_variable, bool use_grad);

void display_variable(const variable_t* variable);
void display_variable_with_gradient(const variable_t* variable);

static inline tensor_entry_t get_entry(const variable_t* variable, tensor_size_t index){
    return _tensor_get_entry(variable->tensor, index);
}

static inline void set_entry(variable_t* variable, tensor_size_t index, tensor_entry_t value){
    _tensor_set_entry(variable->tensor, index, value);
}

static inline void set_to_scalar_value(variable_t* variable, tensor_entry_t value){
    _tensor_set_to_scalar_value(variable->tensor, value);
}

#endif // VARIABLE_H