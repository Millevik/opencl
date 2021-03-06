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

#ifndef CAF_OPENCL_ACTOR_FACADE_HPP
#define CAF_OPENCL_ACTOR_FACADE_HPP

#include <ostream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "caf/all.hpp"

#include "caf/intrusive_ptr.hpp"

#include "caf/detail/int_list.hpp"
#include "caf/detail/type_list.hpp"
#include "caf/detail/limited_vector.hpp"

#include "caf/opencl/global.hpp"
#include "caf/opencl/command.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/opencl_err.hpp"
#include "caf/opencl/spawn_config.hpp"

namespace caf {
namespace opencl {

class manager;

template <class List>
struct function_sig_from_outputs;

template <class... Ts>
struct function_sig_from_outputs<detail::type_list<Ts...>> {
  using type = std::function<message (Ts&...)>;
};

template <class T, class List>
struct command_sig_from_outputs;

template <class T, class... Ts>
struct command_sig_from_outputs<T, detail::type_list<Ts...>> {
  using type = command<T, Ts...>;
};

template <class... Ts>
class actor_facade : public monitorable_actor {
public:
  using arg_types = detail::type_list<Ts...>;
  using unpacked_types = typename detail::tl_map<arg_types, extract_type>::type;

  using input_wrapped_types =
    typename detail::tl_filter<arg_types, is_input_arg>::type;
  using input_types =
    typename detail::tl_map<input_wrapped_types, extract_type>::type;
  using input_mapping = std::function<optional<message> (message&)>;

  using output_wrapped_types =
    typename detail::tl_filter<arg_types, is_output_arg>::type;
  using output_types =
    typename detail::tl_map<output_wrapped_types, extract_type>::type;
  using output_mapping = typename function_sig_from_outputs<output_types>::type;

  typename detail::il_indices<arg_types>::type indices;

  using evnt_vec = std::vector<cl_event>;
  using args_vec = std::vector<mem_ptr>;
  using size_vec = std::vector<size_t>;

  using command_type =
    typename command_sig_from_outputs<actor_facade, output_types>::type;

  const char* name() const override {
    return "OpenCL actor";
  }


  static actor create(actor_config actor_cfg, const program& prog,
                      const char* kernel_name, const spawn_config& spawn_cfg,
                      input_mapping map_args, output_mapping map_result,
                      Ts&&... xs) {
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
      return make_actor<actor_facade, actor>(sys.next_actor_id(), sys.node(),
                                             &sys, std::move(actor_cfg),
                                             prog, kernel, spawn_cfg,
                                             std::move(map_args),
                                             std::move(map_result),
                                             std::forward_as_tuple(xs...));
    }
    return make_actor<actor_facade, actor>(sys.next_actor_id(), sys.node(),
                                           &sys, std::move(actor_cfg),
                                           prog, itr->second, spawn_cfg,
                                           std::move(map_args),
                                           std::move(map_result),
                                           std::forward_as_tuple(xs...));
  }

  void enqueue(strong_actor_ptr sender, message_id mid, message content,
               execution_unit*) override {
    CAF_PUSH_AID(id());
    CAF_LOG_TRACE("");
    if (map_args_) {
      auto mapped = map_args_(content);
      if (! mapped) {
        return;
      }
      content = std::move(*mapped);
    }
    if (! content.match_elements(input_types{})) {
      return;
    }
    auto hdl = std::make_tuple(sender, mid.response_id());
    evnt_vec events;
    args_vec input_buffers;
    args_vec output_buffers;
    size_vec result_sizes;
    add_kernel_arguments(events, input_buffers, output_buffers,
                         result_sizes, content, indices);
    auto cmd = make_counted<command_type>(std::move(hdl),
                                          actor_cast<strong_actor_ptr>(this),
                                          std::move(events),
                                          std::move(input_buffers),
                                          std::move(output_buffers),
                                          std::move(result_sizes),
                                          std::move(content));
    cmd->enqueue();
  }

  void enqueue(mailbox_element_ptr ptr, execution_unit* eu) override {
    CAF_ASSERT(ptr != nullptr);
    CAF_LOG_TRACE(CAF_ARG(*ptr));
    enqueue(ptr->sender, ptr->mid, ptr->move_content_to_message(), eu);
  }

  actor_facade(actor_config actor_cfg,
               const program& prog, kernel_ptr kernel,
               spawn_config  spawn_cfg,
               input_mapping map_args, output_mapping map_result,
               std::tuple<Ts...> xs)
      : monitorable_actor(actor_cfg),
        kernel_(std::move(kernel)),
        program_(prog.program_),
        context_(prog.context_),
        queue_(prog.queue_),
        spawn_cfg_(std::move(spawn_cfg)),
        map_args_(std::move(map_args)),
        map_results_(std::move(map_result)),
        argument_types_(std::move(xs)) {
    CAF_LOG_TRACE(CAF_ARG(this->id()));
    default_output_size_ = std::accumulate(spawn_cfg_.dimensions().begin(),
                                           spawn_cfg_.dimensions().end(),
                                           size_t{1},
                                           std::multiplies<size_t>{});
  }

  void add_kernel_arguments(evnt_vec&, args_vec&, args_vec&, size_vec&,
                            message&, detail::int_list<>) {
    // nop
  }

  /// the separation into input and output is required, because we need to
  /// access the output arguments later on, but only keep the ptrs to the input
  /// to prevent them from being deleted before our operation finished
  template <long I, long... Is>
  void add_kernel_arguments(evnt_vec& events, args_vec& input_buffers,
                            args_vec& output_buffers, size_vec& sizes,
                            message& msg, detail::int_list<I, Is...>) {
    create_buffer<I>(std::get<I>(argument_types_), events, sizes,
                     input_buffers, output_buffers, msg);
    add_kernel_arguments(events, input_buffers, output_buffers, sizes, msg,
                         detail::int_list<Is...>{});
  }

  template <long I, class T>
  void create_buffer(const in<T>&, evnt_vec& events, size_vec&,
                     args_vec& input_buffers, args_vec&, message& msg) {
    using container_type = typename detail::tl_at<unpacked_types, I>::type;
    using value_type = typename container_type::value_type;
    auto& value = msg.get_as<container_type>(I);
    auto size = value.size();
    size_t buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_WRITE}, buffer_size, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, cl_bool{CL_FALSE},
                                 cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr tmp;
    tmp.reset(buffer, false);
    input_buffers.push_back(tmp);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&input_buffers.back()));
  }

  template <long I, class T>
  void create_buffer(const in_out<T>&, evnt_vec& events, size_vec& sizes,
                     args_vec&, args_vec& output_buffers, message& msg) {
    using container_type = typename detail::tl_at<unpacked_types, I>::type;
    using value_type = typename container_type::value_type;
    auto& value = msg.get_as<container_type>(I);
    auto size = value.size();
    size_t buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_WRITE}, buffer_size, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, cl_bool{CL_FALSE},
                                 cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr tmp;
    tmp.reset(buffer, false);
    output_buffers.push_back(tmp);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&output_buffers.back()));
    sizes.push_back(size);
  }

  template <long I, class T>
  void create_buffer(const out<T>& wrapper, evnt_vec&, size_vec& sizes,
                     args_vec&, args_vec& output_buffers, message& msg) {
    using container_type = typename detail::tl_at<unpacked_types, I>::type;
    using value_type = typename container_type::value_type;
    auto size = get_size_for_argument(wrapper, msg, default_output_size_);
    auto buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_WRITE}, buffer_size, nullptr);
    mem_ptr tmp;
    tmp.reset(buffer, false);
    output_buffers.push_back(tmp);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&output_buffers.back()));
    sizes.push_back(size);
  }

  template <class Fun>
  size_t get_size_for_argument(Fun& f, message& m, size_t default_size) {
    auto size = f(m);
    return  size && (*size > 0) ? *size : default_size;
  }

  kernel_ptr kernel_;
  program_ptr program_;
  context_ptr context_;
  command_queue_ptr queue_;
  spawn_config spawn_cfg_;
  input_mapping map_args_;
  output_mapping map_results_;
  std::tuple<Ts...> argument_types_;
  size_t default_output_size_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_ACTOR_FACADE_HPP
