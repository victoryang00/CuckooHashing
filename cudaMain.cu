#include "cudaMain.h"
#include "cuckoo.h"

using namespace std;

int cudaMain(int argc, char **argv) {
    size_t* a[100];
    rand_gen(*a,sizeof(a));
    cout << a[0] <<endl;
    cout << "Hello" << endl;
    return 0;
}
