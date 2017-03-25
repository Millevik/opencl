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
  using args_vec = std::vector<mem_ptr>;
  using size_vec = std::vector<size_t>;
  using outp_tup = typename tuple_type_of<output_types>::type;

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
      return make_actor<opencl_actor, actor>(sys.next_actor_id(), sys.node(),
                                             &sys, std::move(actor_cfg),
                                             prog, kernel, spawn_cfg,
                                             std::move(map_args),
                                             std::move(map_result),
                                             std::forward_as_tuple(xs...));
    }
    return make_actor<opencl_actor, actor>(sys.next_actor_id(), sys.node(),
                                           &sys, std::move(actor_cfg),
                                           prog, itr->second, spawn_cfg,
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
      if (!mapped)
        return;
      content = std::move(*mapped);
    }
    if (!content.match_elements(input_types{})) {
      CAF_LOG_ERROR("Message types do not match the expected signature.");
      return;
    }
    auto hdl = std::make_tuple(sender, mid.response_id());
    evnt_vec events;
    args_vec input_buffers;
    args_vec output_buffers;
    args_vec scratch_buffers;
    size_vec result_sizes;
    outp_tup output_tuple;
    add_kernel_arguments(events,          // accumulate events for execution
                         input_buffers,   // opencl buffers included in in msg
                         output_buffers,  // opencl buffers included in out msg
                         scratch_buffers, // opencl only used here
                         output_tuple,    // tuple to save the output values
                         result_sizes,    // size of buffers to read back
                         content,         // message content
                         indices);        // enable extraction of types from msg
    auto cmd = make_counted<command_type>(
      std::move(hdl),
      actor_cast<strong_actor_ptr>(this),
      std::move(events),
      std::move(input_buffers),
      std::move(output_buffers),
      std::move(scratch_buffers),
      std::move(result_sizes),
      std::move(content),
      std::move(output_tuple),
      config_
    );
    cmd->enqueue();
  }

  void enqueue(mailbox_element_ptr ptr, execution_unit* eu) override {
    CAF_ASSERT(ptr != nullptr);
    CAF_LOG_TRACE(CAF_ARG(*ptr));
    enqueue(ptr->sender, ptr->mid, ptr->move_content_to_message(), eu);
  }

  opencl_actor(actor_config actor_cfg,
               const program& prog, kernel_ptr kernel,
               spawn_config spawn_cfg,
               input_mapping map_args, output_mapping map_result,
               std::tuple<Ts...> xs)
      : monitorable_actor(actor_cfg),
        kernel_(std::move(kernel)),
        program_(prog.program_),
        context_(prog.context_),
        queue_(prog.queue_),
        config_(std::move(spawn_cfg)),
        map_args_(std::move(map_args)),
        map_results_(std::move(map_result)),
        kernel_signature_(std::move(xs)) {
    CAF_LOG_TRACE(CAF_ARG(this->id()));
    default_output_size_ = std::accumulate(config_.dimensions().begin(),
                                           config_.dimensions().end(),
                                           size_t{1},
                                           std::multiplies<size_t>{});
  }

  void add_kernel_arguments(evnt_vec&, args_vec&, args_vec&, args_vec&,
                            outp_tup&, size_vec&, message&,
                            detail::int_list<>) {
    // nop
  }

  /// The separation into input, output and scratch buffers is required to
  /// access the related memory handles later on. The scratch and input handles
  /// are saved to prevent deletion before the kernel finished execution.
  template <long I, long... Is>
  void add_kernel_arguments(evnt_vec& events, args_vec& input_buffers,
                            args_vec& output_buffers, args_vec& scratch_buffers,
                            outp_tup& output_tuple, size_vec& sizes,
                            message& msg, detail::int_list<I, Is...>) {
    using arg_type = typename caf::detail::tl_at<processing_list,I>::type;
    // TODO: In case of mem_refs, should we check if device used for execution
    //       is the same and should we try to transfer memory in such cases?
    create_buffer<I, arg_type::in_pos, arg_type::out_pos>(
      std::get<I>(kernel_signature_), events, sizes, input_buffers,
      output_buffers, scratch_buffers, output_tuple, msg
    );
    add_kernel_arguments(events, input_buffers, output_buffers,
                         scratch_buffers, output_tuple, sizes,
                         msg, detail::int_list<Is...>{});
  }

  // Two functions to handle `in` arguments: val and mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in<T, val>&, evnt_vec& events, size_vec&,
                     args_vec& input_buffers, args_vec&, args_vec&,
                     outp_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& value = msg.get_as<container_type>(InPos);
    auto size = value.size();
    size_t buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, buffer_size, nullptr);
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

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in<T, mref>&, evnt_vec& events, size_vec&,
                     args_vec&, args_vec&, args_vec&, outp_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto mem = msg.get_as<container_type>(InPos);
    auto event = mem.take_event();
    if (event != nullptr)
      events.push_back(event);
    CAF_ASSERT(mem.location() == placement::global_mem);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&mem.get()));
  }

  // Four functions to handle `in_out` arguments:
  //    val->val, val->mref, mref->val, mref->mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,val,val>&, evnt_vec& events,
                     size_vec& sizes, args_vec&, args_vec& output_buffers,
                     args_vec&, outp_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& value = msg.get_as<container_type>(InPos);
    auto size = value.size();
    size_t buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, buffer_size, nullptr);
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

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,val,mref>&, evnt_vec& events, size_vec&,
                     args_vec&, args_vec&, args_vec&, outp_tup& output_tuple,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = std::vector<value_type>;
    auto& value = msg.get_as<container_type>(InPos);
    auto size = value.size();
    size_t buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE, buffer_size, nullptr);
    auto event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                 queue_.get(), buffer, cl_bool{CL_FALSE},
                                 cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr mem;
    mem.reset(buffer, false);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(mem.get()));
    std::get<OutPos>(output_tuple) = mem_ref<T>{size, placement::global_mem,
                                                queue_, mem, CL_MEM_READ_WRITE,
                                                nullptr, false, none};
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,mref,val>&, evnt_vec& events,
                     size_vec& sizes, args_vec&, args_vec& output_buffers,
                     args_vec&, outp_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto mem = msg.get_as<container_type>(InPos);
    auto event = mem.take_event();
    if (event != nullptr)
      events.push_back(event);
    CAF_ASSERT(mem.location() == placement::global_mem);
    output_buffers.push_back(mem.get());
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&output_buffers.back()));
    sizes.push_back(sizeof(value_type) * mem.size());
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const in_out<T,mref,mref>&, evnt_vec& events, size_vec&,
                     args_vec&, args_vec&, args_vec&, outp_tup& output_tuple,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    using container_type = mem_ref<value_type>;
    auto mem = msg.get_as<container_type>(InPos);
    auto event = mem.take_event();
    if (event != nullptr)
      events.push_back(event);
    CAF_ASSERT(mem.location() == placement::global_mem);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
                     sizeof(cl_mem), static_cast<void*>(&mem.get()));
    std::get<OutPos>(output_tuple) = mem;
  }

  // Two functions to handle `out` arguments: val and mref

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const out<T,val>& wrapper, evnt_vec&, size_vec& sizes,
                     args_vec&, args_vec& output_buffers, args_vec&, outp_tup&,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto size = get_size_for_argument(wrapper, msg, default_output_size_);
    auto buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                        buffer_size, nullptr);
    mem_ptr tmp;
    tmp.reset(buffer, false);
    output_buffers.push_back(tmp);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&output_buffers.back()));
    sizes.push_back(size);
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const out<T,mref>& wrapper, evnt_vec&, size_vec&,
                     args_vec&, args_vec&, args_vec&, outp_tup& output_tuple,
                     message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto size = get_size_for_argument(wrapper, msg, default_output_size_);
    auto buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                        buffer_size, nullptr);
    mem_ptr mem;
    mem.reset(buffer, false);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(mem.get()));
    std::get<OutPos>(output_tuple) = mem_ref<T>{
      size, placement::global_mem, queue_, mem,
      CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
      nullptr, false, none
    };
  }

  // One function to handle `scratch` buffers

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const scratch<T>& wrapper, evnt_vec&, size_vec&,
                     args_vec&, args_vec&, args_vec& scratch_buffers,
                     outp_tup&, message& msg) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto size = get_size_for_argument(wrapper, msg, default_output_size_);
    auto buffer_size = sizeof(value_type) * size;
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        buffer_size, nullptr);
    mem_ptr tmp;
    tmp.reset(buffer, false);
    scratch_buffers.push_back(tmp);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             sizeof(cl_mem), static_cast<void*>(&scratch_buffers.back()));
  }

  // One functions to handle `local` arguments

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const local<T>& wrapper, evnt_vec&, size_vec&,
                     args_vec&, args_vec&, args_vec&, outp_tup&,
                     message& msg) {
    using value_type  = typename detail::tl_at<unpacked_types, I>::type;
    auto size = wrapper(msg);
    auto buffer_size = sizeof(value_type) * size;
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             buffer_size, nullptr);
  }

  // Two functions to handle `priv` arguments: val and hidden

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const priv<T, val>&, evnt_vec&, size_vec&,
                     args_vec&, args_vec&, args_vec&, message& msg,
                     outp_tup&) {
    using value_type = typename detail::tl_at<unpacked_types, I>::type;
    auto& value = msg.get_as<value_type>(InPos);
    auto value_size = sizeof(value_type);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             value_size, static_cast<void*>(&value));
  }

  template <long I, int InPos, int OutPos, class T>
  void create_buffer(const priv<T, hidden>& wrapper, evnt_vec&, size_vec&,
                     args_vec&, args_vec&, args_vec&, message& msg,
                     outp_tup&) {
    auto value_size = sizeof(T);
    auto value = wrapper(msg);
    v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), static_cast<unsigned>(I),
             value_size, static_cast<void*>(&value));
  }

  /// Helper function to calculate the elements in a buffer from in and out
  /// argument wrappers.
  template <class Fun>
  size_t get_size_for_argument(Fun& f, message& m, size_t default_size) {
    auto size = f(m);
    return  size && (*size > 0) ? *size : default_size;
  }

  kernel_ptr kernel_;
  program_ptr program_;
  context_ptr context_;
  command_queue_ptr queue_;
  spawn_config config_;
  input_mapping map_args_;
  output_mapping map_results_;
  std::tuple<Ts...> kernel_signature_;
  size_t default_output_size_;
};

} // namespace opencl
} // namespace caf
#endif // CAF_OPENCL_OPENCL_ACTOR_HPP

