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


#include "mem/ruby/network/garnet2.0/SwitchAllocator.hh"

#include "debug/RubyNetwork.hh"
#include "debug/Arbitor.hh"
#include "mem/ruby/network/garnet2.0/GarnetNetwork.hh"
#include "mem/ruby/network/garnet2.0/InputUnit.hh"
#include "mem/ruby/network/garnet2.0/OutputUnit.hh"
#include "mem/ruby/network/garnet2.0/Router.hh"

#include "mem/protocol/LLCResponseMsg.hh"
#include "debug/Frame.hh"

SwitchAllocator::SwitchAllocator(Router *router)
    : Consumer(router)
{
    m_router = router;
    m_num_vcs = m_router->get_num_vcs();
    m_vc_per_vnet = m_router->get_vc_per_vnet();
    alg = ROUND_ROBIN;

    m_input_arbiter_activity = 0;
    m_output_arbiter_activity = 0;
}

void
SwitchAllocator::init()
{
    m_input_unit = m_router->get_inputUnit_ref();
    m_output_unit = m_router->get_outputUnit_ref();

    m_num_inports = m_router->get_num_inports();
    m_num_outports = m_router->get_num_outports();
    m_round_robin_inport.resize(m_num_outports);
    m_round_robin_invc.resize(m_num_inports);
    m_port_requests.resize(m_num_outports);
    m_vc_winners.resize(m_num_outports);

    for (int i = 0; i < m_num_inports; i++) {
        m_round_robin_invc[i] = 0;
    }

    for (int i = 0; i < m_num_outports; i++) {
        m_port_requests[i].resize(m_num_inports);
        m_vc_winners[i].resize(m_num_inports);

        m_round_robin_inport[i] = 0;

        for (int j = 0; j < m_num_inports; j++) {
            m_port_requests[i][j] = false; // [outport][inport]
        }
    }
}

/*
 * The wakeup function of the SwitchAllocator performs a 2-stage
 * seperable switch allocation. At the end of the 2nd stage, a free
 * output VC is assigned to the winning flits of each output port.
 * There is no separate VCAllocator stage like the one in garnet1.0.
 * At the end of this function, the router is rescheduled to wakeup
 * next cycle for peforming SA for any flits ready next cycle.
 */

void
SwitchAllocator::wakeup()
{

    if (alg == ROUND_ROBIN) {
      arbitrate_inports(); // First stage of allocation

      arbitrate_outports(); // Second stage of allocation
    }
    else{
      unified_arbitrate();
    }

    clear_request_vector();
    check_for_wakeup();
}

/*
 * One-stage arbitration algorithm for global aging
 * 
 */
void SwitchAllocator::unified_arbitrate() {
  
  // For each output port, collect information from all input ports and 
  // VCs, send the information into the model and get predictions

  // Pre-compute this so we don't have to recompute every time
  int num_inputs = m_num_inports * m_num_vcs;

  // The final decisions
  std::vector<int> output_port_winners(m_num_outports, invalid_choice);
  std::vector<bool> inport_used(m_num_inports, false);
  
  // Loop through all output ports
  for (int outport = 0; outport < m_num_outports; outport ++ ) {
    // Allocate space for prediction and logging
    std::vector<bool> useful_for_this_port(num_inputs, false);
    std::vector<float> tmp_global_age(num_inputs, 0.0f);
    // Loop through all inport ports, all input VCs, and get the features
    for (int inport = 0; inport < m_num_inports; inport++) {
      // Get the input unit
      auto input_unit = m_input_unit[inport];
      // Go through all virtual channels
      for (int invc = 0; invc < m_num_vcs; invc++){
        // Compute the index
        int idx = inport * m_num_vcs + invc;
        // If this channel needs arbitration, go on
        if (input_unit->need_stage(invc, SA_, m_router->curCycle())) {
          // check if the flit in this InputVC is allowed to be sent
          // send_allowed conditions described in that function.
          int target_outport = input_unit->get_outport(invc);
          int target_outvc = input_unit->get_outvc(invc);
          if (target_outport == outport){
            bool make_request = send_allowed(inport, invc, target_outport, target_outvc);
            if (make_request) {
              flit *t_flit = input_unit->peekTopFlit(invc);
              tmp_global_age[idx] = (m_router->curCycle() - t_flit->get_enqueue_time()) / global_age_norm_factor; 
              useful_for_this_port[idx] = true;
            }
          }
        }
      }
    }
    // Now we should have all the features from each VC
    // with the ones unrelated to the current outport masked

    // If our agent has more than one choice, then refer to the model
    // Otherwise, we can tell right now
    int winner = invalid_choice;
    
    bool easy_choice = check_trivial_cases(useful_for_this_port, winner);
    if(easy_choice) {
      if(winner >-1){
        int inport_tmp = winner/m_num_vcs;
        if (inport_used[inport_tmp]){
          winner = -1;
        }
        else{
          inport_used[inport_tmp] = true;
        }
      }
    }
    else {
      // Choose the best legal result
      winner = choose_best_result(tmp_global_age, useful_for_this_port,inport_used);
    }
    int to_be_sent = 0;
    for(int i=0; i<useful_for_this_port.size(); i++){
      to_be_sent += useful_for_this_port[i];
    }
    DPRINTF(Arbitor, "[GAArbitor] Router %d "
        "compute best invc %d with value %f at outport %d "
        "with %d valid invc "
        "cycle: %lld\n",
        m_router->get_id(), winner, tmp_global_age[winner],
        m_router->getPortDirectionName(
          m_output_unit[outport]->get_direction()),
        to_be_sent,
        m_router->curCycle());
    // Store the winner
    output_port_winners[outport] = winner;
  }
  
  // Actually do the arbitration
  arbitrate_outports_with_winners(output_port_winners);

}

// Check for trivial cases to reduce the number of useless samples
bool SwitchAllocator::check_trivial_cases(
    const std::vector<bool>& useful_for_this_port,
    int& winner) {
  int useful_cnt = 0;
  for (int i = 0; i < useful_for_this_port.size(); i ++ ) {
    if (useful_for_this_port[i]) {
      useful_cnt += 1;
      winner = i;
    }
  }
  if (useful_cnt <= 1) {
    return true;
  } else {
    winner = invalid_choice;
    return false;
  }
}

/* 
 * Choose the best legal result from the ML predictor's scores
 */
int SwitchAllocator::choose_best_result(
    const std::vector<float>& scores, 
    const std::vector<bool>& useful,
    std::vector<bool>& inport_used) {
  // Count the number of actual useful inputs
  int useful_cnt = 0;
  for (int i = 0; i < useful.size(); i ++ )
    useful_cnt += useful[i];
  // If nothing is useful, return -1 (invalid choice)
  if (useful_cnt == 0)
    return invalid_choice;

  float best_score = -100000.0f; // should be small enough
  int best_choice = -1;

  // Choose between random choice and selecting the best
  int rand_num = rand() % 10000;
  if (rand_num < int(10000 * rand_ratio)) {
    // Make a random choice
    int chosen_idx = rand() % useful_cnt;
    int cnter = 0;
    for (int i = 0; i < scores.size(); i ++ ) {
      if (useful[i]) {
        if (cnter == chosen_idx) {
          best_choice = i;
          break;
        } else {
          cnter += 1;
        }
      }
    }
  } else {
    // Not messing with the iterators and stuff
    for (int i = 0; i < scores.size(); i ++ ) {
      if ((scores[i] > best_score) && (useful[i]) && (!inport_used[i/m_num_vcs])) {
        best_score = scores[i];
        best_choice = i;
      }
    }
  }
  //assert(best_choice != invalid_choice);
  if (best_choice >-1){
    inport_used[best_choice/m_num_vcs]=true;
  }
  return best_choice;
}

// send flits to output ports based on the arbitration result 
void SwitchAllocator::arbitrate_outports_with_winners(
    const std::vector<int>& output_port_winners) {
  // We are done with arbitration, get the winners' flits
  // Now there are a set of input vc requests for output vcs.
  // Again do round robin arbitration on these requests
  // Independent arbiter at each output port
  for (int outport = 0; outport < m_num_outports; outport++) {
    // Get the winner
    int winner = output_port_winners[outport];
    // Skip if we don't have any accesses targeting this port at all
    if (winner != invalid_choice) {
      // Compute the actual inport and in-vc of the winner
      int inport = winner / m_num_vcs;
      int invc = winner % m_num_vcs;

      // The rest should be the same with arbitrate_outports()...

      // Get the input and output units
      auto output_unit = m_output_unit[outport];
      auto input_unit = m_input_unit[inport];
      // grant this outport to this inport
      int outvc = input_unit->get_outvc(invc);
      if (outvc == -1) {
        // VC Allocation - select any free VC from outport
        outvc = vc_allocate(outport, inport, invc);
      }

      // remove flit from Input VC
      flit *t_flit = input_unit->getTopFlit(invc);

      DPRINTF(Arbitor, "[GAArbitor] SwitchAllocator at Router %d "
        "granted outvc %d at outport %d "
        "to invc %d at inport %d to flit %s at "
        "cycle: %lld\n",
        m_router->get_id(), outvc,
        m_router->getPortDirectionName(
          output_unit->get_direction()),
        invc,
        m_router->getPortDirectionName(
          input_unit->get_direction()),
        *t_flit,
        m_router->curCycle());


      t_flit->set_outport(outport);
      t_flit->set_vc(outvc);
      output_unit->decrement_credit(outvc);
      t_flit->advance_stage(ST_, m_router->curCycle());
      m_router->grant_switch(inport, t_flit);
      m_output_arbiter_activity++;

      if ((t_flit->get_type() == TAIL_) || (t_flit->get_type() == HEAD_TAIL_)) {
        // This Input VC should now be empty
        assert(!(input_unit->isReady(invc, m_router->curCycle())));
        // Free this VC
        input_unit->set_vc_idle(invc, m_router->curCycle());
        // Send a credit back
        // along with the information that this VC is now idle
        input_unit->increment_credit(invc, true, m_router->curCycle());
      } else {
        // Send a credit back
        // but do not indicate that the VC is idle
        input_unit->increment_credit(invc, false, m_router->curCycle());
      }

      // remove this request
      m_port_requests[outport][inport] = false;

      // Not updating any of the round-robin stuff because we are not using it 
    }
  }
}

/*
 * SA-I (or SA-i) loops through all input VCs at every input port,
 * and selects one in a round robin manner.
 *    - For HEAD/HEAD_TAIL flits only selects an input VC whose output port
 *     has at least one free output VC.
 *    - For BODY/TAIL flits, only selects an input VC that has credits
 *      in its output VC.
 * Places a request for the output port from this input VC.
 */

void
SwitchAllocator::arbitrate_inports()
{
    // Select a VC from each input in a round robin manner
    // Independent arbiter at each input port
    for (int inport = 0; inport < m_num_inports; inport++) {
        int invc = m_round_robin_invc[inport];

        for (int invc_iter = 0; invc_iter < m_num_vcs; invc_iter++) {

            if (m_input_unit[inport]->need_stage(invc, SA_,
                m_router->curCycle())) {

                // This flit is in SA stage

                int  outport = m_input_unit[inport]->get_outport(invc);
                int  outvc   = m_input_unit[inport]->get_outvc(invc);

                // check if the flit in this InputVC is allowed to be sent
                // send_allowed conditions described in that function.
                bool make_request =
                    send_allowed(inport, invc, outport, outvc);

                if (make_request) {

                    // if do make the request see if any other vcs could have gone
                    int invc2 = 0;
                    for (int invc_iter2 = 0; invc_iter2 < m_num_vcs; invc_iter2++) {
                        if (invc2 != invc) {
                            if (m_input_unit[inport]->need_stage(invc2, SA_, m_router->curCycle())) { // is this an issue? whats this checking?
                                // int  outport2 = m_input_unit[inport]->get_outport(invc2);
                                // int  outvc2   = m_input_unit[inport]->get_outvc(invc2);
                                // bool make_request2 =
                                    // send_allowed(inport, invc2, outport2, outvc2);

                                // if (make_request2) {
                                    m_router->updateVcsRouterStall(inport);
                                    break;
                                // }

                                
                            }
                        }

                        invc2++;
                        if (invc2 >= m_num_vcs)
                            invc2 = 0;
                    }

                    m_input_arbiter_activity++;
                    m_port_requests[outport][inport] = true;
                    m_vc_winners[outport][inport]= invc;

                    // Update Round Robin pointer to the next VC
                    m_round_robin_invc[inport] = invc + 1;
                    if (m_round_robin_invc[inport] >= m_num_vcs)
                        m_round_robin_invc[inport] = 0;

                    
                    break; // got one vc winner for this port
                }
                // does nt seem like major contrib
                else {
                    m_router->updateInRouterStall(inport);
                }
            }

            invc++;
            if (invc >= m_num_vcs)
                invc = 0;
        }
    }
}

/*
 * SA-II (or SA-o) loops through all output ports,
 * and selects one input VC (that placed a request during SA-I)
 * as the winner for this output port in a round robin manner.
 *      - For HEAD/HEAD_TAIL flits, performs simplified outvc allocation.
 *        (i.e., select a free VC from the output port).
 *      - For BODY/TAIL flits, decrement a credit in the output vc.
 * The winning flit is read out from the input VC and sent to the
 * CrossbarSwitch.
 * An increment_credit signal is sent from the InputUnit
 * to the upstream router. For HEAD_TAIL/TAIL flits, is_free_signal in the
 * credit is set to true.
 */

void
SwitchAllocator::arbitrate_outports()
{
    // Now there are a set of input vc requests for output vcs.
    // Again do round robin arbitration on these requests
    // Independent arbiter at each output port
    for (int outport = 0; outport < m_num_outports; outport++) {
        int inport = m_round_robin_inport[outport];

        for (int inport_iter = 0; inport_iter < m_num_inports;
                 inport_iter++) {

            // inport has a request this cycle for outport
            if (m_port_requests[outport][inport]) {

                // grant this outport to this inport
                int invc = m_vc_winners[outport][inport];

                int outvc = m_input_unit[inport]->get_outvc(invc);
                if (outvc == -1) {
                    // VC Allocation - select any free VC from outport
                    outvc = vc_allocate(outport, inport, invc);
                }

                // remove flit from Input VC
                flit *t_flit = m_input_unit[inport]->getTopFlit(invc);

                DPRINTF(RubyNetwork, "SwitchAllocator at Router %d "
                                     "granted outvc %d at outport %d "
                                     "to invc %d at inport %d to flit %s at "
                                     "time: %lld\n",
                        m_router->get_id(), outvc,
                        m_router->getPortDirectionName(
                            m_output_unit[outport]->get_direction()),
                        invc,
                        m_router->getPortDirectionName(
                            m_input_unit[inport]->get_direction()),
                            *t_flit,
                        m_router->curCycle());

                // Update outport field in the flit since this is
                // used by CrossbarSwitch code to send it out of
                // correct outport.
                // Note: post route compute in InputUnit,
                // outport is updated in VC, but not in flit
                t_flit->set_outport(outport);

                // set outvc (i.e., invc for next hop) in flit
                // (This was updated in VC by vc_allocate, but not in flit)
                t_flit->set_vc(outvc);

                // decrement credit in outvc
                m_output_unit[outport]->decrement_credit(outvc);

                // flit ready for Switch Traversal
                t_flit->advance_stage(ST_, m_router->curCycle());
                m_router->grant_switch(inport, t_flit);
                m_output_arbiter_activity++;

                if ((t_flit->get_type() == TAIL_) ||
                    t_flit->get_type() == HEAD_TAIL_) {

                    // This Input VC should now be empty
                    assert(!(m_input_unit[inport]->isReady(invc,
                        m_router->curCycle())));

                    // Free this VC
                    m_input_unit[inport]->set_vc_idle(invc,
                        m_router->curCycle());

                    // Send a credit back
                    // along with the information that this VC is now idle
                    m_input_unit[inport]->increment_credit(invc, true,
                        m_router->curCycle());
                } else {
                    // Send a credit back
                    // but do not indicate that the VC is idle
                    m_input_unit[inport]->increment_credit(invc, false,
                        m_router->curCycle());
                }

                // remove this request
                m_port_requests[outport][inport] = false;

                // Update Round Robin pointer
                m_round_robin_inport[outport] = inport + 1;
                if (m_round_robin_inport[outport] >= m_num_inports)
                    m_round_robin_inport[outport] = 0;

                m_router->updateRouterDecision(inport, outport);
                // check if any other ports wanted to use, and count as stall if cant
                int inport2 = 0;
                for (int inport_iter2 = 0; inport_iter2 < m_num_inports; inport_iter2++) {
                    if (m_port_requests[outport][inport2] && inport != inport2) {
                        m_router->updateOutRouterStall(inport2, outport);
                    }

                    inport2++;
                    if (inport2 >= m_num_inports)
                        inport2 = 0;
                }


                break; // got a input winner for this outport
            }

            inport++;
            if (inport >= m_num_inports)
                inport = 0;
        }
    }
}

/*
 * A flit can be sent only if
 * (1) there is at least one free output VC at the
 *     output port (for HEAD/HEAD_TAIL),
 *  or
 * (2) if there is at least one credit (i.e., buffer slot)
 *     within the VC for BODY/TAIL flits of multi-flit packets.
 * and
 * (3) pt-to-pt ordering is not violated in ordered vnets, i.e.,
 *     there should be no other flit in this input port
 *     within an ordered vnet
 *     that arrived before this flit and is requesting the same output port.
 */

bool
SwitchAllocator::send_allowed(int inport, int invc, int outport, int outvc)
{
    // Check if outvc needed
    // Check if credit needed (for multi-flit packet)
    // Check if ordering violated (in ordered vnet)

    int vnet = get_vnet(invc);
    bool has_outvc = (outvc != -1);
    bool has_credit = false;

    if (!has_outvc) {

        // needs outvc
        // this is only true for HEAD and HEAD_TAIL flits.

        if (m_output_unit[outport]->has_free_vc(vnet)) {

            has_outvc = true;

            // each VC has at least one buffer,
            // so no need for additional credit check
            has_credit = true;
        }
    } else {
        has_credit = m_output_unit[outport]->has_credit(outvc);
    }

    // cannot send if no outvc or no credit.
    if (!has_outvc || !has_credit)
        return false;


    // protocol ordering check
    if ((m_router->get_net_ptr())->isVNetOrdered(vnet)) {

        // enqueue time of this flit
        Cycles t_enqueue_time = m_input_unit[inport]->get_enqueue_time(invc);

        // check if any other flit is ready for SA and for same output port
        // and was enqueued before this flit
        int vc_base = vnet*m_vc_per_vnet;
        for (int vc_offset = 0; vc_offset < m_vc_per_vnet; vc_offset++) {
            int temp_vc = vc_base + vc_offset;
            if (m_input_unit[inport]->need_stage(temp_vc, SA_,
                                                 m_router->curCycle()) &&
               (m_input_unit[inport]->get_outport(temp_vc) == outport) &&
               (m_input_unit[inport]->get_enqueue_time(temp_vc) <
                    t_enqueue_time)) {
                return false;
            }
        }
    }

    return true;
}

// Assign a free VC to the winner of the output port.
int
SwitchAllocator::vc_allocate(int outport, int inport, int invc)
{
    // Select a free VC from the output port
    int outvc = m_output_unit[outport]->select_free_vc(get_vnet(invc));

    // has to get a valid VC since it checked before performing SA
    assert(outvc != -1);
    m_input_unit[inport]->grant_outvc(invc, outvc);
    return outvc;
}

// Wakeup the router next cycle to perform SA again
// if there are flits ready.
void
SwitchAllocator::check_for_wakeup()
{
    Cycles nextCycle = m_router->curCycle() + Cycles(1);

    for (int i = 0; i < m_num_inports; i++) {
        for (int j = 0; j < m_num_vcs; j++) {
            if (m_input_unit[i]->need_stage(j, SA_, nextCycle)) {
                // m_router->schedule_wakeup(Cycles(1)); // TODO bad
                m_router->schedule_wakeup((Tick)m_router->clockEdge(Cycles(0)) + (Tick)1);
                return;
            }
        }
    }
}





int
SwitchAllocator::get_vnet(int invc)
{
    int vnet = invc/m_vc_per_vnet;
    assert(vnet < m_router->get_num_vnets());
    return vnet;
}


// Clear the request vector within the allocator at end of SA-II.
// Was populated by SA-I.
void
SwitchAllocator::clear_request_vector()
{
    for (int i = 0; i < m_num_outports; i++) {
        for (int j = 0; j < m_num_inports; j++) {
            m_port_requests[i][j] = false;
        }
    }
}

void
SwitchAllocator::resetStats()
{
    m_input_arbiter_activity = 0;
    m_output_arbiter_activity = 0;
}
