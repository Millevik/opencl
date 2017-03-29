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

#ifndef CAF_OPENCL_OPENCL_ACTOR_HPP
#define CAF_OPENCL_OPENCL_ACTOR_HPP

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
#include "caf/opencl/mem_ref.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/opencl_err.hpp"
#include "caf/opencl/spawn_config.hpp"


namespace caf {
namespace opencl {

class manager;

// signature for the function that is applied to output arguments
template <class List>
struct output_function_sig;

template <class... Ts>
struct output_function_sig<detail::type_list<Ts...>> {
  using type = std::function<message (Ts&...)>;
};

// convert to mem_ref
template <class T>
struct to_mem_ref {
  using type = mem_ref<T>;
};

template <class T>
struct to_mem_ref<std::vector<T>> {
  using type = mem_ref<T>;
};

template <class T>
struct to_mem_ref<T*> {
  using type = mem_ref<T>;
};

// derive signature of the command that handles the kernel execution
template <class T, class List>
struct command_sig;

template <class T, class... Ts>
struct command_sig<T, detail::type_list<Ts...>> {
  using type = command<T, Ts...>;
};

// derive type for a tuple matching the arguments as mem_refs
template <class List>
struct tuple_type_of;

template <class... Ts>
struct tuple_type_of<detail::type_list<Ts...>> {
  using type = std::tuple<Ts...>;
};

template <class... Ts>
class opencl_actor : public monitorable_actor {
public:
  using arg_types = detail::type_list<Ts...>;
  using unpacked_types = typename detail::tl_map<arg_types, extract_type>::type;

  using input_wrapped_types =
    typename detail::tl_filter<arg_types, is_input_arg>::type;
  using input_types =
    typename detail::tl_map<input_wrapped_types, extract_input_type>::type;
  using input_mapping = std::function<optional<message> (message&)>;

  using output_wrapped_types =
    typename detail::tl_filter<arg_types, is_output_arg>::type;
  using output_types =
    typename detail::tl_map<output_wrapped_types, extract_output_type>::type;
  using output_mapping = typename output_function_sig<output_types>::type;

  using processing_list = typename cl_arg_info_list<arg_types>::type;

  using command_type = typename command_sig<opencl_actor, output_types>::type;

  typename detail::il_indices<arg_types>::type indices;

  using evnt_vec = std::vector<cl_event>;
  using mem_vec = std::vector<mem_ptr>;
  using len_vec = std::vector<size_t>;
  using out_tup = typename tuple_type_of<output_types>::type;

  const char* name() const override {
    return "OpenCL actor";
  }

  static actor create(actor_config actor_conf, const program& prog,
                      const char* kernel_name, const spawn_config& spawn_conf,
                      input_mapping map_args, output_mapping map_result,
                      Ts&&... xs) {
    if (spawn_conf.dimensions().empty()) {
      auto str = "OpenCL kernel needs at least 1 global dimension.";
      CAF_LOG_ERROR(str);
      throw std::runtime_error(str);
    }
    auto check_vec = [&](const dim_vec& vec, const char* name) {
      if (! vec.empty() && vec.size() != spawn_conf.dimensions().size()) {
        std::ostringstream oss;
        oss << name << " vector is not empty, but "
            << "its size differs from global dimensions vector's size";
        CAF_LOG_ERROR(CAF_ARG(oss.str()));
        throw std::runtime_error(oss.str());
      }
    };
    check_vec(spawn_conf.offsets(), "offsets");
    check_vec(spawn_conf.local_dimensions(), "local dimensions");
    auto& sys = actor_conf.host->system();
    auto itr = prog.available_kernels_.find(kernel_name);
    if (itr == prog.available_kernels_.end()) {
      kernel_ptr kernel;
      kernel.reset(v2get(CAF_CLF(clCreateKernel), prog.program_.get(),
                                 kernel_name),
                   false);
      return make_actor<opencl_actor, actor>(sys.next_actor_id(), sys.node(),
                                             &sys, std::move(actor_conf),
                                             prog, kernel, spawn_conf,
                                             std::move(map_args),
                                             std::move(map_result),
                                             std::forward_as_tuple(xs...));
    }
    return make_actor<opencl_actor, actor>(sys.next_actor_id(), sys.node(),
                                           &sys, std::move(actor_conf),
                                           prog, itr->second, spawn_conf,
                                           std::move(map_args),
                                           std::move(map_result),
                                           std::forward_as_tuple(xs...));
  }

  void enqueue(strong_actor_ptr sender, message_id mid,
               message content, execution_unit*) override {
    CAF_PUSH_AID(id());
    CAF_LOG_TRACE("");
    if (map_args_) {
      auto mapped = map_args_(content);
      if (!mapped) {
        CAF_LOG_ERROR("Mapping argumentes failed.");
        return;
      }
      content = std::move(*mapped);
    }
    if (!content.match_elements(input_types{})) {
      CAF_LOG_ERROR("Message types do not match the expected signature.");
      return;
    }
    auto hdl = std::make_tuple(sender, mid.response_id());
    evnt_vec events;
    mem_vec input_buffers;
    mem_vec output_buffers;
    mem_vec scratch_buffers;
    len_vec result_lengths;
    out_tup result;
    add_kernel_arguments(events,          // accumulate events for execution
                         input_buffers,   // opencl buffers included in in msg
                         output_buffers,  // opencl buffers included in out msg
                         scratch_buffers, // opencl only used here
                         result,          // tuple to save the output values
                         result_lengths,  // size of buffers to read back
                         content,         // message content
                         indices);        // enable extraction of types from msg
    auto cmd = make_counted<command_type>(
      std::move(hdl),
      actor_cast<strong_actor_ptr>(this),
      std::move(events),
      std::move(input_buffers),
      std::move(output_buffers),
      std::move(scratch_buffers),
      std::move(result_lengths),
      std::move(content),
      std::move(result),
      config_
    );
    cmd->enqueue();
  }

  void enqueue(mailbox_element_ptr ptr, execution_unit* eu) override {
    CAF_ASSERT(ptr != nullptr);
    CAF_LOG_TRACE(CAF_ARG(*ptr));
    enqueue(ptr->sender, ptr->mid, ptr->move_content_to_message(), eu);
  }

  opencl_actor(actor_config actor_conf,
               const program& prog, kernel_ptr kernel,
               spawn_config spawn_conf,
               input_mapping map_args, output_mapping map_result,
               std::tuple<Ts...> xs)
      : monitorable_actor(actor_conf),
        kernel_(std::move(kernel)),
        program_(prog.program_),
        context_(prog.context_),
        queue_(prog.queue_),
        config_(std::move(spawn_conf)),
        map_args_(std::move(map_args)),
        map_results_(std::move(map_result)),
        kernel_signature_(std::move(xs)) {
    CAF_LOG_TRACE(CAF_ARG(this->id()));
    default_length_ = std::accumulate(config_.dimensions().begin(),
                                      config_.dimensions().end(),
                                      size_t{1},
                                      std::multiplies<size_t>{});
  }

  void add_kernel_arguments(evnt_vec&, mem_vec&, mem_vec&, mem_vec&,
                            out_tup&, len_vec&, message&,
                            detail::int_list<>) {
    // nop
  }

  /// The separation into input, output and scratch buffers is required to
  /// access the related memory handles later on. The scratch and input handles
  /// are saved to prevent deletion before the kernel finished execution.
  template <long I, long... Is>
  void add_kernel_arguments(evnt_vec& events, mem_vec& inputs, mem_vec& outputs,
                            mem_vec& scratch, out_tup& result, len_vec& lengths,
                            message& msg, detail::int_list<I, Is...>) {
    using arg_type = typename caf::detail::tl_at<processing_list,I>::type;
    // TODO: In case of mem_refs, should we check if device used for execution
    //       is the same and should we try to transfer memory in such cases?
    create_buffer<I, arg_type::in_pos, arg_type::out_pos>(
      std::get<I>(kernel_signature_), events, lengths, inputs,
      outputs, scratch, result, msg
    );
    add_kernel_arguments(events, inputs, outputs,
                         scratch, result, lengths,
                         msg, detail::int_list<Is...>{});
  }

  // Two functions to handle `in` arguments: val and mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in<T, val>&, evnt_vec& events, len_vec&,
                     mem_vec& inputs, mem_vec&, mem_vec&, out_tup&,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& container = msg.get_as<container_type>(InPos);
    auto len = container.size();
    size_t num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, num_bytes, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, CL_FALSE,
                                 0u, num_bytes, container.data());
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    events.push_back(event);
    inputs.emplace_back(buffer, false);
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in<T, mref>&, evnt_vec& events, len_vec&, mem_vec&,
                     mem_vec&, mem_vec&, out_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto container = msg.get_as<container_type>(InPos);
    CAF_ASSERT(container.location() == placement::global_mem);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&container.get()));
    auto event = container.take_event();
    if (event != nullptr)
      events.push_back(event);
  }

  // Four functions to handle `in_out` arguments:
  //    val->val, val->mref, mref->val, mref->mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,val,val>&, evnt_vec& events,
                     len_vec& lengths, mem_vec&, mem_vec& outputs,
                     mem_vec&, out_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& container = msg.get_as<container_type>(InPos);
    auto len = container.size();
    size_t num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, num_bytes, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, CL_FALSE,
                                 0u, num_bytes, container.data());
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    lengths.push_back(len);
    events.push_back(event);
    outputs.emplace_back(buffer, false);
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,val,mref>&, evnt_vec& events, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup& result,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& container = msg.get_as<container_type>(InPos);
    auto len = container.size();
    size_t num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, num_bytes, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, CL_FALSE,
                                 0u, num_bytes, container.data());
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    events.push_back(event);
    std::get<OutPos>(result) = mem_ref<value_type>{
      len, placement::global_mem, queue_, mem_ptr{buffer, false},
      CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
      nullptr, false, none
    };
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,mref,val>&, evnt_vec& events,
                     len_vec& lengths, mem_vec&, mem_vec& outputs,
                     mem_vec&, out_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto container = msg.get_as<container_type>(InPos);
    CAF_ASSERT(container.location() == placement::global_mem);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&container.get()));
    auto event = container.take_event();
    if (event != nullptr)
      events.push_back(event);
    lengths.push_back(container.size());
    outputs.push_back(container.get());
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,mref,mref>&, evnt_vec& events, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup& result,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto container = msg.get_as<container_type>(InPos);
    CAF_ASSERT(container.location() == placement::global_mem);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
                     sizeof(cl_mem), static_cast<const void*>(&container.get()));
    auto event = container.take_event();
    if (event != nullptr)
      events.push_back(event);
    std::get<OutPos>(result) = container;
  }

  // Two functions to handle `out` arguments: val and mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const out<T,val>& wrapper, evnt_vec&, len_vec& lengths,
                     mem_vec&, mem_vec& outputs, mem_vec&, out_tup&,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto len = argument_length(wrapper, msg, default_length_);
    auto num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                        num_bytes, nullptr);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    outputs.emplace_back(buffer, false);
    lengths.push_back(len);
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const out<T,mref>& wrapper, evnt_vec&, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup& result,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto len = argument_length(wrapper, msg, default_length_);
    auto num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                        num_bytes, nullptr);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    std::get<OutPos>(result) = mem_ref<value_type>{
      len, placement::global_mem, queue_, {buffer, false},
      CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, nullptr, false, none
    };
  }

  // One function to handle `scratch` buffers

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const scratch<T>& wrapper, evnt_vec&, len_vec&,
                     mem_vec&, mem_vec&, mem_vec& scratch,
                     out_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto len = argument_length(wrapper, msg, default_length_);
    auto num_bytes = sizeof(value_type) * len;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        num_bytes, nullptr);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<const void*>(&buffer));
    scratch.emplace_back(buffer, false);
  }

  // One functions to handle `local` arguments

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const local<T>& wrapper, evnt_vec&, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup&,
                     message& msg) {
    using value_type  = typename detail::tl_at<unpacked_types, I>::type;
    auto len = wrapper(msg);
    auto num_bytes = sizeof(value_type) * len;
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             num_bytes, nullptr);
  }

  // Two functions to handle `priv` arguments: val and hidden

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const priv<T, val>&, evnt_vec&, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto value_size = sizeof(value_type);
    auto& value = msg.get_as<value_type>(InPos);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             value_size, static_cast<const void*>(&value));
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const priv<T, hidden>& wrapper, evnt_vec&, len_vec&,
                     mem_vec&, mem_vec&, mem_vec&, out_tup&, message& msg) {
    auto value_size = sizeof(T);
    auto value = wrapper(msg);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             value_size, static_cast<const void*>(&value));
  }

  /// Helper function to calculate the elements in a buffer from in and out
  /// argument wrappers.
  template <class Fun>
  size_t argument_length(Fun& f, message& m, size_t fallback) {
    auto length = f(m);
    return  length && (*length > 0) ? *length : fallback;
  }

  kernel_ptr kernel_;
  program_ptr program_;
  context_ptr context_;
  command_queue_ptr queue_;
  spawn_config config_;
  input_mapping map_args_;
  output_mapping map_results_;
  std::tuple<Ts...> kernel_signature_;
  size_t default_length_;
};

} // namespace opencl
} // namespace caf
#endif // CAF_OPENCL_OPENCL_ACTOR_HPP

