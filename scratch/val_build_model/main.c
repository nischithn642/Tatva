
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "model_info.h"

extern int32_t tvmgen_default_run(void* input_ids, void* attention_mask, void* token_type_ids, void* output);

static inline uint64_t read_cycles(void) {
    uint64_t cycles;
#if __riscv_xlen == 64
    asm volatile ("rdcycle %0" : "=r" (cycles));
#else
    uint32_t cycle_h, cycle_l, cycle_h_again;
    do {
        asm volatile ("rdcycleh %0" : "=r" (cycle_h));
        asm volatile ("rdcycle %0" : "=r" (cycle_l));
        asm volatile ("rdcycleh %0" : "=r" (cycle_h_again));
    } while (cycle_h != cycle_h_again);
    cycles = (((uint64_t)cycle_h) << 32) | cycle_l;
#endif
    return cycles;
}

static int local_errno;
int* __errno(void) {
    return &local_errno;
}

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

void sbi_print_uint(uint64_t val) {
    char buf[32];
    int i = 0;
    if (val == 0) {
        sbi_print("0");
        return;
    }
    while (val > 0) {
        buf[i++] = '0' + (val % 10);
        val /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        sbi_putchar(buf[j]);
    }
}

void sbi_print_float(float val) {
    if (val < 0) {
        sbi_print("-");
        val = -val;
    }
    uint64_t integer_part = (uint64_t)val;
    sbi_print_uint(integer_part);
    sbi_print(".");
    float fraction = val - (float)integer_part;
    for (int i = 0; i < 6; i++) {
        fraction *= 10.0f;
        uint32_t digit = (uint32_t)fraction;
        sbi_putchar('0' + digit);
        fraction -= (float)digit;
    }
}

void sbi_shutdown(void) {
    register unsigned long a0 asm("a0") = 0;
    register unsigned long a1 asm("a1") = 0;
    register unsigned long a6 asm("a6") = 0;
    register unsigned long a7 asm("a7") = 0x53525354;
    asm volatile ("ecall" : : "r"(a0), "r"(a1), "r"(a6), "r"(a7) : "memory");
}

static int64_t input_ids[32];
static int64_t attention_mask[32];
static int64_t token_type_ids[32];
static float output_logits[OUTPUT_SIZE];

int main(void) {
    sbi_print("\n=== Starting Latency Test inside QEMU ===\n");
    
    // Initialize dummy inputs
    for (int i = 0; i < 32; i++) {
        input_ids[i] = i % 5;
        attention_mask[i] = 1;
        token_type_ids[i] = 0;
    }
    
    // Warm-up runs
    for (int i = 0; i < 2; i++) {
        tvmgen_default_run(input_ids, attention_mask, token_type_ids, output_logits);
    }
    
    // Timed runs
    for (int i = 0; i < 5; i++) {
        uint64_t start = read_cycles();
        tvmgen_default_run(input_ids, attention_mask, token_type_ids, output_logits);
        uint64_t end = read_cycles();
        sbi_print("RUN_CYCLES: ");
        sbi_print_uint(end - start);
        sbi_print("\n");
    }
    
    // Print first 5 logits for correctness verification
    sbi_print("FIRST_LOGITS: ");
    for (int i = 0; i < 5 && i < OUTPUT_SIZE; i++) {
        sbi_print_float(output_logits[i]);
        sbi_print(" ");
    }
    sbi_print("\n");
    
    sbi_print("=== Latency Test Finished ===\n");
    sbi_shutdown();
    while (1);
    return 0;
}
