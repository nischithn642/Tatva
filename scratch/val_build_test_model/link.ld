
OUTPUT_ARCH( "riscv" )
ENTRY( _start )

MEMORY
{
  ram (wxa) : ORIGIN = 0x80200000, LENGTH = 128M
}

SECTIONS
{
  . = 0x80200000;
  
  .text : {
    *(.text.init)
    *(.text .text.*)
  } > ram

  .rodata : {
    *(.rodata .rodata.*)
  } > ram

  .data : {
    *(.data .data.*)
  } > ram

  .bss : {
    . = ALIGN(8);
    _bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > ram

  . = ALIGN(16);
  _stack_bottom = .;
  . += 0x10000; /* 64KB stack */
  _stack_top = .;
}
