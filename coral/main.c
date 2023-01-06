#include "tensor.h"
#include "variable.h"
#include "assert.h"
#include <stdio.h>

int main() {
    variable_t* x = variable_new(3, 2, 2, 4);
    variable_t* y = variable_new(1, 4);
    variable_set_to_scalar_value(x, 10);
    variable_set_to_scalar_value(y, 2);
    variable_t* z = variable_add(x, y);
    variable_display(x);
    variable_display(y);
    variable_display(z);
    DEBUG_ASSERT(0, "Oh no, %d\n", 3);
}