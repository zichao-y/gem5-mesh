#==============================================================================
# brg_scratchpad.py
#==============================================================================
# Python configuration file for BRG scratchpad system
#
# Authors: Tuan Ta
#          Philip Bedoukian
# Date  : 19/07/09

import optparse
import sys
import os
import math

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal, warn

from caches import *

addToPath('../')

from ruby import Ruby
from common import Options
from topologies.Crossbar import *
from network.Network import *


from common import SimpleOpts


from math import log

#------------------------------------------------------------------------------
# Get workload process
#------------------------------------------------------------------------------

def get_processes(options):
    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = options.cmd.split(';')
    if options.input != "":
        inputs = options.input.split(';')
    if options.output != "":
        outputs = options.output.split(';')
    if options.errout != "":
        errouts = options.errout.split(';')
    if options.options != "":
        pargs = options.options.split(';')

    idx = 0
    for wrkld in workloads:
        process = Process(pid = 100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()

        if options.env:
            with open(options.env, 'r') as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    return multiprocesses

parser = optparse.OptionParser()
parser.add_option("-c", "--cmd", default="",
                      help="The binary to run in syscall emulation mode.")
parser.add_option("-o", "--options", default="",
                      help="""The options to pass to the binary, use " "
                              around the entire string""")
parser.add_option("-e", "--env", default="",
                      help="Initialize workload environment from text file.")
parser.add_option("-i", "--input", default="",
                      help="Read stdin from a file.")
parser.add_option("--output", default="",
                  help="Redirect stdout to a file.")
parser.add_option("--errout", default="",
                  help="Redirect stderr to a file.")
parser.add_option("--l1d_size", type="string", default="4kB")
parser.add_option("--l1i_size", type="string", default="4kB")  
parser.add_option("--l2_size", type="string", default="1MB")     
       
(options, args) = parser.parse_args()
# Do not support multi-process simulation
#process = get_processes(options)[0]
isa = str(m5.defines.buildEnv['TARGET_ISA']).lower()
thispath = os.path.dirname(os.path.realpath(__file__))
binary = os.path.join(thispath, '../../',
                      'programs-spad/loadtest/spmm')
#------------------------------------------------------------------------------
# Construct CPUs
#------------------------------------------------------------------------------
# create the system we are going to simulate
system = System()

# Set the clock fequency of the system (and all of its children)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# Set up the system
system.mem_mode = 'timing'               # Use timing accesses
system.mem_ranges = [AddrRange('8192MB')] # Create an address range

# Create a simple CPU
system.cpu = IOCPU()

# Create an L1 instruction and data cache
system.cpu.icache = L1ICache(options)
system.cpu.dcache = L1DCache(options)

# Connect the instruction and data caches to the CPU
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# Create a memory bus, a coherent crossbar, in this case
system.l2bus = L2XBar()

# Hook the CPU ports up to the l2bus
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

# Create an L2 cache and connect it to the l2bus
system.l2cache = L2Cache(options)
system.l2cache.connectCPUSideBus(system.l2bus)

# Create a memory bus
system.membus = SystemXBar()

# Connect the L2 cache to the membus
system.l2cache.connectMemSideBus(system.membus)

# create the interrupt controller for the CPU
system.cpu.createInterruptController()

# For x86 only, make sure the interrupts are connected to the memory
# Note: these are directly connected to the memory bus and are not cached
if m5.defines.buildEnv['TARGET_ISA'] == "x86":
    system.cpu.interrupts[0].pio = system.membus.master
    system.cpu.interrupts[0].int_master = system.membus.slave
    system.cpu.interrupts[0].int_slave = system.membus.master

# Connect the system up to the membus
system.system_port = system.membus.slave

# Create a DDR3 memory controller
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.master

# Create a process for a simple "Hello World" application
process = Process()
# Set the command
# cmd is a list which begins with the executable (like argv)
process.cmd = [binary]
# Set the cpu to use the process as its workload and create thread contexts
system.cpu.workload = process
system.cpu.createThreads()
# set up the root SimObject and start the simulation
root = Root(full_system = False, system = system)
# instantiate all of the objects we've created above
m5.instantiate()



print("Beginning simulation!")
exit_event = m5.simulate()
print('Exiting @ tick %i because %s' % (m5.curTick(), exit_event.getCause()))
print('Exit code %i' % exit_event.getCode())
