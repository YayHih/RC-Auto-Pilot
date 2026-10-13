/* Linker script for the emulator build: flash at 0, RAM at 0x20000000 (the ASR6502 has 16 KB;
 * the emulator maps 64 KB and the build reports how much the flight controller really needs). */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 128K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}
ENTRY(fc_arm_tick)
EXTERN(fc_arm_tick fc_arm_init fc_arm_defaults)
SECTIONS
{
    .text : { *(.text*) *(.rodata*) . = ALIGN(4); } > FLASH
    .data : { __data_start = .; *(.data*) . = ALIGN(4); __data_end = .; } > RAM
    .bss (NOLOAD) : { __bss_start = .; *(.bss*) *(COMMON) . = ALIGN(4); __bss_end = .; } > RAM
    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment) *(.ARM.attributes) }
}
