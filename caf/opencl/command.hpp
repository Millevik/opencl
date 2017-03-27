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

#ifndef CAF_OPENCL_COMMAND_HPP
#define CAF_OPENCL_COMMAND_HPP

#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#include "caf/logger.hpp"
#include "caf/actor_cast.hpp"
#include "caf/abstract_actor.hpp"
#include "caf/response_promise.hpp"

#include "caf/detail/scope_guard.hpp"

#include "caf/opencl/global.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/opencl_err.hpp"

namespace caf {
namespace opencl {

template <class Facade, class... Ts>
class command : public ref_counted {
public:
  command(std::tuple<strong_actor_ptr,message_id> handle,
               strong_actor_ptr actor_facade,
               std::vector<cl_event> events,
               std::vector<mem_ptr> input_bufs,
               std::vector<mem_ptr> output_bufs,
               std::vector<mem_ptr> scratch_bufs,
               std::vector<size_t> result_sizes,
               message msg,
               std::tuple<Ts...> output_tuple,
               spawn_config config)
      : result_sizes_(std::move(result_sizes)),
        handle_(std::move(handle)),
        opencl_actor_(std::move(actor_facade)),
        mem_in_events_(std::move(events)),
        input_buffers_(std::move(input_bufs)),
        output_buffers_(std::move(output_bufs)),
        scratch_buffers_(std::move(scratch_bufs)),
        results_(output_tuple),
        msg_(std::move(msg)),
        config_(std::move(config)) {
    // nop
  }

  ~command() override {
    for (auto& e : mem_in_events_) {
      v1callcl(CAF_CLF(clReleaseEvent),e);
    }
    for (auto& e : mem_out_events_) {
      v1callcl(CAF_CLF(clReleaseEvent),e);
    }
  }

  void enqueue() {
    // Errors in this function can not be handled by opencl_err.hpp
    // because they require non-standard error handling
    CAF_LOG_TRACE("command::enqueue()");
    this->ref(); // reference held by the OpenCL comand queue
    cl_event event_k;
    auto data_or_nullptr = [](const dim_vec& vec) {
      return vec.empty() ? nullptr : vec.data();
    };
    auto actor_facade =
      static_cast<Facade*>(actor_cast<abstract_actor*>(opencl_actor_));
    // OpenCL expects cl_uint (unsigned int), hence the cast
    cl_int err = clEnqueueNDRangeKernel(
      actor_facade->queue_.get(), actor_facade->kernel_.get(),
      static_cast<cl_uint>(config_.dimensions().size()),
      data_or_nullptr(config_.offsets()),
      data_or_nullptr(config_.dimensions()),
      data_or_nullptr(config_.local_dimensions()),
      static_cast<cl_uint>(mem_in_events_.size()),
      (mem_in_events_.empty() ? nullptr : mem_in_events_.data()), &event_k
    );
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clEnqueueNDRangeKernel: "
                    << CAF_ARG(get_opencl_error(err)));
      clReleaseEvent(event_k);
      this->deref();
      return;
    }
    size_t pos = 0;
    enqueue_read_buffers(event_k, pos, detail::get_indices(results_));
    cl_event marker;
#if defined(__APPLE__)
    err = clEnqueueMarkerWithWaitList(
      actor_facade->queue_.get(),
      static_cast<cl_uint>(mem_out_events_.size()),
      mem_out_events_.data(), &marker
    );
#else
    err = clEnqueueMarker(actor_facade->queue_.get(), &marker);
#endif
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clSetEventCallback: " << CAF_ARG(get_opencl_error(err)));
      clReleaseEvent(marker);
      clReleaseEvent(event_k);
      this->deref(); // callback is not set
      return;
    }
    err = clSetEventCallback(marker, CL_COMPLETE,
                             [](cl_event, cl_int, void* data) {
                               auto cmd = reinterpret_cast<command*>(data);
                               cmd->handle_results();
                               cmd->deref();
                             },
                             this);
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clSetEventCallback: " << CAF_ARG(get_opencl_error(err)));
      clReleaseEvent(marker);
      clReleaseEvent(event_k);
      this->deref(); // callback is not set
      return;
    }
    err = clFlush(actor_facade->queue_.get());
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clFlush: " << CAF_ARG(get_opencl_error(err)));
    }
    mem_out_events_.push_back(std::move(event_k));
    mem_out_events_.push_back(std::move(marker));
  }

private:
  template <long I, class T>
  void enqueue_read(std::vector<T>&, cl_event& kernel_done, size_t& pos) {
    auto af = static_cast<Facade*>(actor_cast<abstract_actor*>(opencl_actor_));
    cl_event event;
    auto size = result_sizes_[pos];
    auto buffer_size = sizeof(T) * size;
    std::get<I>(results_).resize(size);
    auto err = clEnqueueReadBuffer(af->queue_.get(), output_buffers_[pos].get(),
                                   CL_FALSE, 0, buffer_size,
                                   std::get<I>(results_).data(), 1,
                                   &kernel_done, &event);
    if (err != CL_SUCCESS) {
      this->deref(); // failed to enqueue command
      throw std::runtime_error("clEnqueueReadBuffer: " + get_opencl_error(err));
    }
    mem_out_events_.push_back(std::move(event));
    pos += 1;
  };

  template <long I, class T>
  void enqueue_read(mem_ref<T>&, cl_event&, size_t&) {
    // nop
  };

  void enqueue_read_buffers(cl_event&, size_t&, detail::int_list<>) {
    // end of recursion
  }

  template <long I, long... Is>
  void enqueue_read_buffers(cl_event& kernel_done, size_t& pos,
                            detail::int_list<I, Is...>) {
    enqueue_read<I>(std::get<I>(results_), kernel_done, pos);
    enqueue_read_buffers(kernel_done, pos, detail::int_list<Is...>{});
  }

  void handle_results() {
    auto actor_facade =
      static_cast<Facade*>(actor_cast<abstract_actor*>(opencl_actor_));
    auto& map_fun = actor_facade->map_results_;
    auto msg = map_fun ? apply_args(map_fun, detail::get_indices(results_),
                                    results_)
                       : message_from_results{}(results_);
    get<0>(handle_)->enqueue(opencl_actor_, get<1>(handle_), std::move(msg),
                             nullptr);
  }

  std::vector<size_t> result_sizes_;
  std::tuple<strong_actor_ptr,message_id> handle_;
  strong_actor_ptr opencl_actor_;
  std::vector<cl_event> mem_in_events_;
  std::vector<cl_event> mem_out_events_;
  std::vector<mem_ptr> input_buffers_;
  std::vector<mem_ptr> output_buffers_;
  std::vector<mem_ptr> scratch_buffers_;
  std::tuple<Ts...> results_;
  message msg_; // required to keep the argument buffers alive (async copy)
  spawn_config config_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_COMMAND_HPP
