#include "tensor.h"
#include "variable.h"
#include "assert.h"
#include <stdbool.h>


void test_variable_equality(){
    printf("Testing variable equality...");
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
    printf("...PASS.\n");
}

void test_variable_add(){
    printf("Testing variable addition...");
    variable_t* x = variable_new(2, 3, 4);
    variable_set_to_scalar_value(x, 1.0);
    variable_t* y = variable_new_like_with_value(x, 3.0);
    variable_t* z_expected = variable_new_like_with_value(x, 4.0);
    variable_t* z_actual = variable_add(x, y);
    NDEBUG_ASSERT(variable_equal(z_expected, z_actual), "Expected does not match actual.");
    printf("PASS.\n");
}

void test_variable_subtract(){
    printf("Testing variable subtraction...");
    variable_t* x = variable_new(3, 3, 4, 5);
    variable_t* y = variable_new(2, 3, 20);
    variable_set_to_scalar_value(x, 4.0);
    variable_set_to_scalar_value(y, 3.0);
    variable_in_place_view_as(x, 2, 3, 20);
    variable_t* z_actual = variable_subtract(x, y);
    variable_t* z_expected = variable_new_like_with_value(x, 1.0);
    NDEBUG_ASSERT(variable_equal(z_expected, z_actual), "Expected does not match actual.");
    printf("PASS.\n");
}

void test_variable_multiply();
void test_varaible_square();
void test_variable_abs();
void test_broadcast();
void test_loss();

void test_backwards();


int main(){
    test_variable_equality();
    test_variable_add();
    test_variable_subtract();
    printf("All tests passed! :D");
    return 0;
}
