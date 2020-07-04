#include "cuckoo.cuh"
#include <chrono>

using namespace std;
#ifndef DEMO
int main(int argc, char **argv) {
   // Experiment 1 - Insertion time v.s. number of keys.
    std::cout << "Experiment 1 -->" << std::endl;
    {
        int size = 0x1 << 25;
        for (int num_funcs = 3; num_funcs <= 4; ++num_funcs) {
            for (int scale = 10; scale <= 24; ++scale) {
                int n = 0x1 << scale;
                int *vals_to_insert = new int[n];
                for (int rep = 0; rep < 5; ++rep) {
                    gen_rnd_input(vals_to_insert, n, 0x1 << 30);
                    std::cout << " t = " << num_funcs << ","
                              << " scale = " << scale << ","
                              << " rep " << rep << ":  ";

                    // Serial.
                    #ifndef C
                    {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                             num_funcs);
                        auto ts = std::chrono::high_resolution_clock::now();
                        int max_levels = 0;
                        for (int i = 0; i < n; ++i) {
                            int levels = hash_table.insert(vals_to_insert[i], 0);
                            if (levels == ERROR_DEPTH) {
                                max_levels = ERROR_DEPTH;
                                break;
                            } else if (levels > max_levels)
                                max_levels = levels;
                        }
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[Serial] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms - ";
                        if (max_levels == ERROR_DEPTH)
                            std::cout << "exceeds " << MAX_DEPTH << " levels | ";
                        else
                            std::cout << " " << max_levels << " rehash(es) | ";
                    }

                    // CUDA.
                    {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                                       num_funcs);
                        auto ts = std::chrono::high_resolution_clock::now();
                        int levels = hash_table.insert(vals_to_insert, n, 0);
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[CUDA] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms - ";
                        if (levels == ERROR_DEPTH)
                            std::cout << "exceeds " << MAX_DEPTH << " levels" << std::endl;
                        else
                            std::cout << " " << levels << " rehash(es)" << std::endl;
                    }
                }
                delete[] vals_to_insert;
            }
        }
    }

    // Experiment 2 - Lookup time v.s. lookup configurations.
    std::cout << "Experiment 2 -->" << std::endl;
    {
        int size = 0x1 << 25, n = 0x1 << 24;
        for (int num_funcs = 3; num_funcs <= 4; ++num_funcs) {
            int *vals_to_insert = new int[n];
            int *vals_to_lookup = new int[n];
            bool *results = new bool[n];
            for (int percent = 0; percent <= 10; ++percent) {
                int bound = ceil((1 - 0.1 * percent) * n);
                for (int rep = 0; rep < 5; ++rep) {
                    gen_rnd_input(vals_to_insert, n, (int) 0x1 << 30);
                    for (int i = 0; i < bound; ++i)
                        vals_to_lookup[i] = vals_to_insert[rand() % n];
                    for (int i = bound; i < n; ++i)
                        vals_to_lookup[i] = (rand() % ((0x1 << 30) - 1)) + 1;
                    std::cout << " t = " << num_funcs << ","
                              << " percentile = " << " " << percent << ","
                              << " rep " << rep << ":  ";

                    // Serial.
                    {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                             num_funcs);
                        for (int i = 0; i < n; ++i)
                            hash_table.insert(vals_to_insert[i], 0);
                        auto ts = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < n; ++i)
                            hash_table.lookup_val(vals_to_lookup[i]);
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[Serial] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms | ";
                    }

                    // CUDA.
                    {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                                       num_funcs);
                        hash_table.insert(vals_to_insert, n, 0);
                        auto ts = std::chrono::high_resolution_clock::now();
                        hash_table.lookup_vals(vals_to_lookup, results, n);
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[CUDA] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms" << std::endl;
                    }
                }
            }
            delete[] vals_to_insert;
            delete[] vals_to_lookup;
            delete[] results;
        }
    }

    // Experiment 3 - Insertion time v.s. size.
    std::cout << "Experiment 3 -->" << std::endl;
    {
        int n = 0x1 << 24;
        int *vals_to_insert = new int[n];
        gen_rnd_input(vals_to_insert, n, 0x1 << 30);
        for (int num_funcs = 3; num_funcs <= 4; ++num_funcs) {
            float ratios[] = {1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01};
            for (int ri = 0; ri < 12; ++ri) {
                int size = ceil(ratios[ri] * n);
                for (int rep = 0; rep < 5; ++rep) {
                    std::cout << " t = " << num_funcs << ","
                              << " ratio = " << std::fixed << std::setprecision(2) << ratios[ri] << ","
                              << " rep " << rep << ":  ";

                    // Serial.
                    if (ri < 8) {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                             num_funcs);
                        auto ts = std::chrono::high_resolution_clock::now();
                        int max_levels = 0;
                        for (int i = 0; i < n; ++i) {
                            int levels = hash_table.insert(vals_to_insert[i], 0);
                            if (levels == ERROR_DEPTH) {
                                max_levels = ERROR_DEPTH;
                                break;
                            } else if (levels > max_levels)
                                max_levels = levels;
                        }
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[Serial] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms - ";
                        if (max_levels == ERROR_DEPTH)
                            std::cout << "exceeds " << MAX_DEPTH << " levels | ";
                        else
                            std::cout << " " << max_levels << " rehash(es) | ";
                    }

                    // CUDA.
                    {
                        CuckooHashing<int> hash_table(size, 4 * ceil(log2((double) n)),
                                                                       num_funcs);
                        auto ts = std::chrono::high_resolution_clock::now();
                        int levels = hash_table.insert(vals_to_insert, n, 0);
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[CUDA] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms - ";
                        if (levels == ERROR_DEPTH)
                            std::cout << "exceeds " << MAX_DEPTH << " levels" << std::endl;
                        else
                            std::cout << " " << levels << " rehash(es)" << std::endl;
                    }
                }
            }
        }
        delete[] vals_to_insert;
    }

    // Experiment 4 - Insertion time v.s. eviction bound.
    std::cout << "Experiment 4 -->" << std::endl;
    {
        int n = 0x1 << 24, size = ceil(1.4 * n);
        int *vals_to_insert = new int[n];
        rand_gen(vals_to_insert, n);
        for (int num_funcs = 3; num_funcs <= 4; ++num_funcs) {
            for (int bound_mul = 1; bound_mul <= 10; ++bound_mul) {
                for (int rep = 0; rep < 5; ++rep) {
                    std::cout << " t = " << num_funcs << ","
                              << " bound = " << " " << bound_mul << " log n,"
                              << " rep " << rep << ":  ";

                    // CUDA.
                    {
                        CuckooHashing<int> hash_table(size, bound_mul * ceil(log2((double) n)),
                                                                       num_funcs);
                        auto ts = std::chrono::high_resolution_clock::now();
                        int levels = hash_table.insert(vals_to_insert, n, 0);
                        auto te = std::chrono::high_resolution_clock::now();
                        std::cout << "[CUDA] " << " "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                                  << " ms - ";
                        if (levels == ERROR_DEPTH)
                            std::cout << "exceeds " << MAX_DEPTH << " levels" << std::endl;
                        else
                            std::cout << " " << levels << " rehash(es)" << std::endl;
                    }
                }
            }
        }
        delete[] vals_to_insert;
    }
    return 0;
}
#endif