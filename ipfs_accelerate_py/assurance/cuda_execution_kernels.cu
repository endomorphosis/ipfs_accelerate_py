/* PCPR-038 live CUDA canary source.

The sealed validation PATH does not include nvcc, and /usr/local/cuda is
not an admitted toolchain. Runtime qualification loads the checked-in PTX
through libcuda. This file documents the kernel semantics; it is not
executed by the live canary.
*/

extern "C" __global__ void addone(const unsigned int* in, unsigned int* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] + 1u;
    }
}

extern "C" __global__ void mix32(unsigned int* acc_out, unsigned int n) {
    unsigned int acc = 0u;
    const unsigned int A = 1103515245u;
    const unsigned int B = 12345u;
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int t = i * A + B;
        acc = acc + (t ^ (acc << 1));
    }
    acc_out[0] = acc;
}

extern "C" __global__ void spin_until_cancel(volatile unsigned int* flag, unsigned int* ticks) {
    unsigned int n = 0u;
    while (*flag == 0u) {
        ++n;
    }
    *ticks = n;
}
