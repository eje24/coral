#ifndef GRAD_H
#define GRAD_H

#include "variable.h"

static inline void increment_ref_count(variable_t* variable){
    variable->grad_meta->ref_count++;
}

static inline void decrement_ref_count(variable_t* variable){
    variable->grad_meta->ref_count--;
}

static inline int get_ref_count(variable_t* variable){
    return variable->grad_meta->ref_count;
}

void set_unary_grad_meta(variable_t* child, variable_t* parent, variable_unary_grad_op_t grad_op);
void set_binary_grad_meta(variable_t* child, variable_t* parent1, variable_t* parent2, variable_binary_grad_op_t grad_op1, variable_binary_grad_op_t grad_op2);

#endif // GRAD_H