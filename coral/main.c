#include "tensor.h"
#include "variable.h"
#include "assert.h"
#include <stdio.h>

int main() {
    variable_t* x = variable_new(3, 2, 2, 4);
    variable_t* y = variable_new(2, 2, 4);
    variable_t* z = variable_new(1, 4);
    variable_set_to_scalar_value(x, 10);
    variable_in_place_apply_index_fn(y, index_identity);
    variable_set_to_scalar_value(z, 2);
    variable_display(x);
    variable_display(y);
    variable_t* z1 = variable_add(x, y);
    variable_t* z2 = variable_subtract(x, y);
    variable_t* z3 = variable_multiply(x, y);
    variable_display(z1);
    variable_display(z2);
    variable_display(z3);
    DEBUG_ASSERT(0, "Oh no, %d\n", 3);
}