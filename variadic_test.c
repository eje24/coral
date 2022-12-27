#include <stdarg.h>
#include <stdio.h>

int add(int n, ...){
    va_list ap;
    va_start(ap, n);
    int sum = 0;
    for(int i=0;i<n;i++){
        sum += va_arg(ap, int);
    }
    va_end(ap);
    return sum;
}

int main() {
    int res = add(6, 1, 2, 3, 5, 6, 7);
    printf("Sum is: %d\n", res);
}