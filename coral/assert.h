#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

// Copyright (c) 2022 MIT License by Tao B. Schardl
#define _DEBUG_ASSERT(PREDICATE, ...)                                    \
  do {                                                                  \
    if (!(PREDICATE)) {                                                 \
      fprintf(stderr,                                                   \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__, \
              __LINE__, __PRETTY_FUNCTION__);                           \
      fprintf(stderr, __VA_ARGS__);                                     \
      abort();                                                          \
    }                                                                   \
  } while (0)

// for general use
#define NDEBUG_ASSERT _DEBUG_ASSERT

#ifndef NDEBUG
#include <stdio.h>
#include <stdlib.h>

// for debug use
#define DEBUG_ASSERT _DEBUG_ASSERT

#else
#define DEBUG_ASSERT(PREDICATE, ...)
#endif
  
#endif // DEBUG_H