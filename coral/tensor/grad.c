#include "grad.h"
#include "variable.h"
#include "debug.h"

// recursively traverse the computation graph backwards
// updating the gradients as we go

void _update_unary_grad(diff_arg_t* arg, variable_t* result){
    variable_unary_grad_op_t gradient_fn = (variable_unary_grad_op_t) (arg->grad_op);
    tensor_t* gradient_update = (*gradient_fn)(arg->arg, result);
    _tensor_add_to_existing(arg->arg->gradient, gradient_update);
    _decrement_ref_count(arg->arg);
}

void _update_binary_grad(diff_arg_t* arg, diff_arg_t* other_arg, variable_t* result){
    variable_binary_grad_op_t gradient_fn = (variable_binary_grad_op_t) (arg->grad_op);
    tensor_t* gradient_update = (*gradient_fn)(arg->arg, other_arg->arg, result);
    _tensor_add_to_existing(arg->arg->gradient, gradient_update);
    _decrement_ref_count(arg->arg);
}

// accumulate gradient updates into argument gradients
static inline void _update_binary_grads(diff_arg_t* left_arg, diff_arg_t* right_arg, variable_t* result){
    _update_binary_grad(left_arg, right_arg, result);
    _update_binary_grad(right_arg, left_arg, result);
}

// accumulates gradient update into argument(s') gradient
// for arguments with ref_count zero, call _backwards an arguments
// so as recurse in a way that respects gradient
// graph's topological ordering
void _backwards(variable_t* root){
    if(root->grad_meta->num_args == 1){
        diff_arg_t* arg = root->grad_meta->args[0];
        _update_unary_grad(arg, root);
        if(_get_ref_count(arg->arg) == 0){
            _backwards(arg->arg);
        } 
    }else if(root->grad_meta->num_args == 2){
        diff_arg_t* arg1 = root->grad_meta->args[0];
        diff_arg_t* arg2 = root->grad_meta->args[1];
        _update_binary_grads(arg1, arg2, root);
        if(_get_ref_count(arg1->arg) == 0){
            _backwards(arg1->arg);
        }
        if(_get_ref_count(arg2->arg) == 0){
            _backwards(arg2->arg);
        }
    }
}

void backwards(variable_t* root){
    NDEBUG_ASSERT(is_scalar(root), "Error: root variable is not a scalar.");
    // set root gradient to 1
    _tensor_set_to_scalar_value(root->gradient, 1);
    _backwards(root);
}

void _unary_set_grad_meta(variable_t* child, variable_t* parent, variable_grad_op_t grad_op){
    diff_arg_t* diff_arg = _new_diff_arg(parent, grad_op);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_args = 1;
    new_grad_meta->args[0] = diff_arg;
    child->grad_meta = new_grad_meta;
    _increment_ref_count(parent);
}

void _binary_set_grad_meta(variable_t* child, variable_t* parent1, variable_t* parent2, variable_binary_grad_op_t grad_op1, variable_binary_grad_op_t grad_op2){
    diff_arg_t* diff_arg1 = _new_diff_arg(parent1, grad_op1);
    diff_arg_t* diff_arg2 = _new_diff_arg(parent2, grad_op2);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_args = 2;
    new_grad_meta->args[0] = diff_arg1;
    new_grad_meta->args[1] = diff_arg2;
    child->grad_meta = new_grad_meta;
    _increment_ref_count(parent1);
    _increment_ref_count(parent2);
}
