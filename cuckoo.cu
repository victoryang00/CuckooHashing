#include <stdio.h>
#include <ctime>
#include <malloc.h>
#include <map>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <ctime>
#include "cuckoo.h"
#include "cudaHeaders.h"

using namespace std;
/* Cuckoo hashing CPU version */




/* Randomize Generation */
void rand_gen(int *vals, const int n) {
    map<int, bool> val_map;
    int i = 0;
    while (i < n) {
        int value = (rand() % (LIMIT - 1)) + 1;
        if (val_map.find(value) != val_map.end()) {
            continue;
        }
        val_map[value] = true;
        vals[i] = value;
        i++;
    }
}

/* Single gpu implementation */

