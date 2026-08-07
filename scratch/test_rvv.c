#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void* memset(void* dest, int c, size_t n) {
    unsigned char* d = dest;
    while (n--) {
        *d++ = (unsigned char)c;
    }
    return dest;
}

void sbi_putchar(char c) {
    register unsigned long a0 asm("a0") = (unsigned long)c;
    register unsigned long a7 asm("a7") = 0x01;
    asm volatile ("ecall" : : "r"(a0), "r"(a7) : "memory");
}

void sbi_print(const char* str) {
    while (*str) {
        sbi_putchar(*str++);
    }
}

void sbi_shutdown(void) {
    register unsigned long a0 asm("a0") = 0;
    register unsigned long a1 asm("a1") = 0;
    register unsigned long a6 asm("a6") = 0;
    register unsigned long a7 asm("a7") = 0x53525354;
    asm volatile ("ecall" : : "r"(a0), "r"(a1), "r"(a6), "r"(a7) : "memory");
}

void rvv_add_f32(const float* a, const float* b, float* c, size_t n) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(c, vc, vl);
    }
}

int main(void) {
    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float c[8] = {0.0f};

    rvv_add_f32(a, b, c, 8);

    if (c[0] == 11.0f && c[7] == 88.0f) {
        sbi_print("RVV_VECTOR_SUCCESS\\n");
    } else {
        sbi_print("RVV_VECTOR_FAIL\\n");
    }

    sbi_shutdown();
    while(1);
    return 0;
}
