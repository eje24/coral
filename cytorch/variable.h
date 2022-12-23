#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <stdbool.h>

// wrapper around tensor
// intended to store metadata necessary for backpropagation
typedef struct {
    tensor_t tensor;
    tensor_t gradient; 
} variable_t;

variable_t _new_variable(tensor_t tensor);
variable_t _new_variable_like(variable_t old_variable);
variable_t _copy_variable(variable_t old_variable);

#endif // VARIABLE_H