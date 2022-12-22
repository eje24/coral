#ifndef DEBUG_H
#define DEBUG_H

#ifndef NDEBUG
#include <stdio.h>
#include <stdlib.h>

// from tbassert.h, authored Tao B. Schardl
#define DEBUG_ASSERT(PREDICATE, ...)                                    \
  do {                                                                  \
    if (!(PREDICATE)) {                                                 \
      fprintf(stderr,                                                   \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__, \
              __LINE__, __PRETTY_FUNCTION__);                           \
      fprintf(stderr, __VA_ARGS__);                                     \
      abort();                                                          \
    }                                                                   \
  } while (0)

#else
#define DEBUG_ASSERT(PREDICATE, ...)
#endif
  
#endif // DEBUG_H