#include "tensor.h"
#include "variable.h"
#include "assert.h"
#include <stdbool.h>


void test_variable_equality(){
    variable_t* x1 = variable_new(2, 3 ,4);
    variable_t* x2 = variable_new(2, 3, 4);
    variable_t* y1 = variable_new(1, 4);
    variable_t* y2 = variable_new(1, 4);
    variable_set_to_scalar_value(x1, 2);
    variable_set_to_scalar_value(y2, 2);
    variable_t* z1 = variable_add(x1, y1);
    variable_t* z2 = variable_add(x2, y2);
    NDEBUG_ASSERT(variable_equal(z1, z2), "Variables z1, z2 should be equal.");
    variable_t* z3 = variable_view_as(z2, 2, 6, 2);
    NDEBUG_ASSERT(variable_alias(z2, z3), "Variables z2, z2 are aliases.");
}

void test_variable_add();
void test_variable_subtract();
void test_variable_multiply();
void test_varaible_square();
void test_variable_abs();

void test_backwards();


int main(){
    test_variable_equality();
    printf("All tests passed! :D");
    return 0;
}
