#include "cudaMain.h"
#include "cuckoo.h"

using namespace std;

int cudaMain(int argc, char **argv) {
    int a[100];
    for (int i=0;i<100;i++){
        a[i]=1;
    }
    rand_gen(a,sizeof(a));

    cout << a[0] <<endl;
    cout << "Hello" << endl;
    return 0;
}
