/*
 * Copyright (c) 2008 Princeton University
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Niket Agarwal
 *          Tushar Krishna
 */


#ifndef __MEM_RUBY_NETWORK_GARNET2_0_SWITCHALLOCATOR_HH__
#define __MEM_RUBY_NETWORK_GARNET2_0_SWITCHALLOCATOR_HH__

#include <iostream>
#include <vector>

#include "mem/ruby/common/Consumer.hh"
#include "mem/ruby/network/garnet2.0/CommonTypes.hh"

typedef enum {
  ROUND_ROBIN = 0,
  RL,
  GLOBAL_AGE,
  LOGIC,
  LOCAL_AGE,
  TREE
} ArbitrationAlg;
const int invalid_choice = -1;
const float global_age_norm_factor = 500.0f;
const float rand_ratio = 0.0f;

class Router;
class InputUnit;
class OutputUnit;

class SwitchAllocator : public Consumer
{
  public:
    SwitchAllocator(Router *router);
    void wakeup();
    void init();
    void clear_request_vector();
    void check_for_wakeup();
    int get_vnet (int invc);
    void print(std::ostream& out) const {};
    void arbitrate_inports();
    void arbitrate_outports();
    void unified_arbitrate();
    bool send_allowed(int inport, int invc, int outport, int outvc);
    int vc_allocate(int outport, int inport, int invc);

    inline double
    get_input_arbiter_activity()
    {
        return m_input_arbiter_activity;
    }
    inline double
    get_output_arbiter_activity()
    {
        return m_output_arbiter_activity;
    }

    void resetStats();
    // Check for trivial cases to reduce the number of useless samples
    bool check_trivial_cases(
      const std::vector<bool>& useful_for_this_port,
      int& winner);
    int choose_best_result(
      const std::vector<float>& scores,
      const std::vector<bool>& useful,
      std::vector<bool>& inport_used);
    // send flits to output ports based on the arbitration result
    void arbitrate_outports_with_winners(const std::vector<int>& output_port_winners);

  private:
    int m_num_inports, m_num_outports;
    int m_num_vcs, m_vc_per_vnet;
    ArbitrationAlg alg;

    double m_input_arbiter_activity, m_output_arbiter_activity;

    Router *m_router;
    std::vector<int> m_round_robin_invc;
    std::vector<int> m_round_robin_inport;
    std::vector<std::vector<bool>> m_port_requests;
    std::vector<std::vector<int>> m_vc_winners; // a list for each outport
    std::vector<InputUnit *> m_input_unit;
    std::vector<OutputUnit *> m_output_unit;
};

#endif // __MEM_RUBY_NETWORK_GARNET2_0_SWITCHALLOCATOR_HH__
