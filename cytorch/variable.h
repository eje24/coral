#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"

typedef struct {
    tensor_t tensor;
    tensor_t gradient; 
} variable_t;

variable_t _new_variable_like(variable_t old_variable);
variable_t _copy_variable(variable_t old_variable);

#endif // VARIABLE_H