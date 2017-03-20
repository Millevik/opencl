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

#ifndef CAF_OPENCL_async_command_HPP
#define CAF_OPENCL_async_command_HPP

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

template <class FacadeType, class... Ts>
class async_command : public ref_counted {
public:
  async_command(std::tuple<strong_actor_ptr,message_id> handle,
                strong_actor_ptr facade,
                std::vector<cl_event> events,
                std::tuple<Ts...> refs)
    : handle_(std::move(handle)),
      actor_facade_(std::move(facade)),
      events_(std::move(events)),
      refs_(std::move(refs)) {
    // nop
  }

  ~async_command() {
    if (execution_)
      clReleaseEvent(execution_);
    for (auto& e : events_)
      clReleaseEvent(e);
  }

  void enqueue() {
    // Errors in this function can not be handled by opencl_err.hpp
    // because they require non-standard error handling
    CAF_LOG_TRACE("command::enqueue()");
    this->ref(); // reference held by the OpenCL command queue
    auto data_or_nullptr = [](const dim_vec& vec) {
      return vec.empty() ? nullptr : vec.data();
    };
    auto facade =
      static_cast<FacadeType*>(actor_cast<abstract_actor*>(actor_facade_));
    auto err = clEnqueueNDRangeKernel(
      facade->queue_.get(), facade->kernel_.get(),
      static_cast<cl_uint>(facade->spawn_cfg_.dimensions().size()),
      data_or_nullptr(facade->spawn_cfg_.offsets()),
      data_or_nullptr(facade->spawn_cfg_.dimensions()),
      data_or_nullptr(facade->spawn_cfg_.local_dimensions()),
      static_cast<cl_uint>(events_.size()),
      (events_.empty() ? nullptr : events_.data()), &execution_
    );
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clEnqueueNDRangeKernel: "
                    << CAF_ARG(get_opencl_error(err)));
      if (execution_)
        clReleaseEvent(execution_);
      this->deref();
      return;
    }
    err = clSetEventCallback(execution_, CL_COMPLETE,
                             [](cl_event, cl_int, void* data) {
                               auto c = reinterpret_cast<async_command*>(data);
                               c->deref();
                             },
                             this);
    if (err != CL_SUCCESS) {
      CAF_LOG_ERROR("clSetEventCallback: " << CAF_ARG(get_opencl_error(err)));
      if (execution_)
        clReleaseEvent(execution_);
      this->deref(); // callback is not set
      return;
    }
    err = clFlush(facade->queue_.get());
    if (err != CL_SUCCESS)
      CAF_LOG_ERROR("clFlush: " << CAF_ARG(get_opencl_error(err)));
    auto msg = msg_adding_event{execution_}(refs_);
    get<0>(handle_)->enqueue(actor_facade_, get<1>(handle_), std::move(msg),
                             nullptr);
  }

private:
  std::tuple<strong_actor_ptr,message_id> handle_;
  strong_actor_ptr actor_facade_;
  std::vector<cl_event> events_;
  std::tuple<Ts...> refs_;
  cl_event execution_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_async_command_HPP
