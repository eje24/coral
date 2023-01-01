#ifndef GRAD_H
#define GRAD_H

#include "variable.h"

static inline _increment_ref_count(variable_t* variable){
    variable->grad_meta->ref_count++;
}

static inline _decrement_ref_count(variable_t* variable){
    variable->grad_meta->ref_count--;
}

static inline int _get_ref_count(variable_t* variable){
    return variable->grad_meta->ref_count;
}

void _unary_set_grad_meta(variable_t* child, variable_t* parent, variable_grad_op_t grad_op);
void _binary_set_grad_meta(variable_t* child, variable_t* parent1, variable_t* parent2, variable_binary_grad_op_t grad_op1, variable_binary_grad_op_t grad_op2);

#endif // GRAD_H