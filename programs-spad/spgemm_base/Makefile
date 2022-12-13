BENCHNAME := spmm
default: $(BENCHNAME)

SIZE_M := 100
SIZE_N := 100
SIZE_K := 16
DENSITY := 0.2

N_ROWS := 8
N_COLS := 8
# Specify number of CPUs. (TODO: rename to NUM_CPUS)
#DEBUG_FLAGS := --debug-flags=Frame # --debug-start=1900830000 #--debug-end=3077250000#--debug-flags=LineTrace #--debug-start=256895000#--debug-flags=Mesh #--debug-file=coraO3_withlinetrace.out.gz --debug-start=3992240000 --debug-end=4113289000#RubyCache,Scratchpad 
N_SPS ?= $$(($(N_ROWS)*$(N_COLS)))
EXTRA_FLAGS ?= -DNO_VEC #-DCACHE_WARM#-DPER_CORE_SIMD #-DVECTOR_LEN=4
HB_ARGS := --options="$(N_COLS) $(N_ROWS)"
CONFIG_ARGS := --num-row=$(N_ROWS) --num-col=$(N_COLS) --doublel2=False --l2-size=128kB --crossbar_bw=16 --cpu-type=IOCPU --spm-size=64kB
# GEM5_ARGS ?= --remote-gdb-port=0 --debug-flags=RiscvVector

# Build the kernel with trilliasm.
#TRILLIASM_KERNEL := gemm_kernel.c
#include ../../trillium/trilliasm.mk

# Standard benchmark compilation.
include ../common/common.mk

