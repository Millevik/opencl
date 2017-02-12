/******************************************************************************
 *                       ____    _    _____                                   *
 *                      / ___|  / \  |  ___|    C++                           *
 *                     | |     / _ \ | |_       Actor                         *
 *                     | |___ / ___ \|  _|      Framework                     *
 *                      \____/_/   \_|_|                                      *
 *                                                                            *
 * Copyright (C) 2011 - 2016                                                  *
 * Dominik Charousset <dominik.charousset (at) haw-hamburg.de>                *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License or  *
 * (at your option) under the terms and conditions of the Boost Software      *
 * License 1.0. See accompanying files LICENSE and LICENSE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#ifndef CAF_OPENCL_ACTOR_FACADE_PHASE_HPP
#define CAF_OPENCL_ACTOR_FACADE_PHASE_HPP

#include "caf/all.hpp"

#include "caf/detail/int_list.hpp"
#include "caf/detail/type_list.hpp"
#include "caf/detail/limited_vector.hpp"

#include "caf/opencl/global.hpp"
#include "caf/opencl/mem_ref.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/opencl_err.hpp"
#include "caf/opencl/spawn_config.hpp"
#include "caf/opencl/phase_command.hpp"

namespace caf {
namespace opencl {

class manager;

// convert to mem_ref
template <class T>
struct as_mem_ref { };

template <class T>
struct as_mem_ref<std::vector<T>> {
  using type = mem_ref<T>;
};

template <class T>
struct as_mem_ref<T*> {
  using type = mem_ref<T>;
};

// derive signature of suitable phase_command
template <class T, class List>
struct phase_command_signature;

template <class T, class... Ts>
struct phase_command_signature<T, detail::type_list<Ts...>> {
  using type = phase_command<T, Ts...>;
};

// derive type for a tuple matching the arguments as mem_refs
template <class List>
struct tuple_mem_ref_type;

template <class... Ts>
struct tuple_mem_ref_type<detail::type_list<Ts...>> {
  using type = std::tuple<Ts...>;
};

template <class... Ts>
class actor_facade_phase : public monitorable_actor {
public:
  using arg_types = detail::type_list<Ts...>;
  using unpacked_types = typename detail::tl_map<arg_types, carr_to_vec>::type;

  using mem_ref_types = typename detail::tl_map<arg_types, as_mem_ref>::type;

  typename detail::il_indices<arg_types>::type indices;

  using evnt_vec = std::vector<cl_event>;
  using args_vec = std::vector<mem_ptr>;
  using size_vec = std::vector<size_t>;

  using command_type =
    typename phase_command_signature<actor_facade_phase, mem_ref_types>::type;

  using tuple_type = 
    typename tuple_mem_ref_type<mem_ref_types>::type;

  const char* name() const override {
    return "OpenCL phase actor";
  }

  static actor create(actor_config actor_cfg, const program& prog,
                      const char* kernel_name, const spawn_config& spawn_cfg) {
    if (spawn_cfg.dimensions().empty()) {
      auto str = "OpenCL kernel needs at least 1 global dimension.";
      CAF_LOG_ERROR(str);
      throw std::runtime_error(str);
    }
    auto check_vec = [&](const dim_vec& vec, const char* name) {
      if (! vec.empty() && vec.size() != spawn_cfg.dimensions().size()) {
        std::ostringstream oss;
        oss << name << " vector is not empty, but "
            << "its size differs from global dimensions vector's size";
        CAF_LOG_ERROR(CAF_ARG(oss.str()));
        throw std::runtime_error(oss.str());
      }
    };
    check_vec(spawn_cfg.offsets(), "offsets");
    check_vec(spawn_cfg.local_dimensions(), "local dimensions");
    auto& sys = actor_cfg.host->system();
    auto itr = prog.available_kernels_.find(kernel_name);
    if (itr == prog.available_kernels_.end()) {
      kernel_ptr kernel;
      kernel.reset(v2get(CAF_CLF(clCreateKernel), prog.program_.get(),
                                 kernel_name),
                   false);
      return make_actor<actor_facade_phase, actor>(sys.next_actor_id(),
                                                   sys.node(), &sys,
                                                   std::move(actor_cfg),
                                                   prog, kernel, spawn_cfg);
    }
    return make_actor<actor_facade_phase, actor>(sys.next_actor_id(),
                                                 sys.node(), &sys, 
                                                 std::move(actor_cfg),
                                                 prog, itr->second, spawn_cfg);
  }

  void enqueue(strong_actor_ptr sender, message_id mid, message content,
               execution_unit*) override {
    CAF_PUSH_AID(id());
    CAF_LOG_TRACE("");
    if (!content.match_elements(mem_ref_types{}))
      return;
    std::vector<cl_event> events;
    tuple_type refs;
    set_kernel_arguments(content, refs, events, indices);
    auto hdl = std::make_tuple(sender, mid.response_id());
    auto cmd = make_counted<command_type>(std::move(hdl),
                                          actor_cast<strong_actor_ptr>(this),
                                          std::move(events),
                                          std::move(refs));
    cmd->enqueue();
  }

  void enqueue(mailbox_element_ptr ptr, execution_unit* eu) override {
    CAF_ASSERT(ptr != nullptr);
    CAF_LOG_TRACE(CAF_ARG(*ptr));
    enqueue(ptr->sender, ptr->mid, ptr->move_content_to_message(), eu);
  }

  actor_facade_phase(actor_config actor_cfg,
                     const program& prog, kernel_ptr kernel,
                     spawn_config  spawn_cfg)
      : monitorable_actor(actor_cfg),
        kernel_(std::move(kernel)),
        program_(prog.program_),
        context_(prog.context_),
        queue_(prog.queue_),
        spawn_cfg_(std::move(spawn_cfg)) {
    CAF_LOG_TRACE(CAF_ARG(this->id()));
  }


  void set_kernel_arguments(message&, tuple_type&, std::vector<cl_event>&,
                            detail::int_list<>) {
    // nop
  }

  template <long I, long... Is>
  void set_kernel_arguments(message& msg, tuple_type& refs,
                            std::vector<cl_event>& events,
                            detail::int_list<I, Is...>) {
    using mem_type = typename detail::tl_at<mem_ref_types, I>::type;
    auto mem = msg.get_as<mem_type>(I);
    events.push_back(mem.event().get());
    get<I>(refs) = mem;
    // TODO: check if device used for execution is the same as for the 
    //       mem_ref, should we try to transfer memory in such cases?
    // TODO: add support for local arguments
    // (require buffer size instead of cl_mem size)
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&mem.get()));
  }

  kernel_ptr kernel_;
  program_ptr program_;
  context_ptr context_;
  command_queue_ptr queue_;
  spawn_config spawn_cfg_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_ACTOR_FACADE_PHASE_HPP
