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

#ifndef CAF_OPENCL_MEM_REF_HPP
#define CAF_OPENCL_MEM_REF_HPP

#include <vector>

#include "caf/sec.hpp"
#include "caf/optional.hpp"
#include "caf/ref_counted.hpp"

#include "caf/opencl/device.hpp"
#include "caf/opencl/smart_ptr.hpp"

namespace caf {
namespace opencl {

enum buffer_type : cl_mem_flags {
  input         = CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
  input_output  = CL_MEM_READ_WRITE,
  output        = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
  scratch_space = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS
};

/// A reference type for buffers on a OpenCL devive
template <class T>
class mem_ref : public ref_counted {
public:
  friend class device;
  template <class Inspector, class U>
  typename Inspector::result_type inspect(Inspector& f, mem_ref<U>& x) {
    return f(meta::type_name("mem_ref"), x.input_size_, x.result_size_,
             x.type_, x.device_, x.event_, x.memory_);
  }

  expected<std::vector<T>> data(optional<size_t> result_size = none) {
    if (result_size && *result_size > input_size_)
      return make_error(sec::runtime_error, "Cannot read more than buffer size.");
    command_queue_ptr queue = device_->queue_;
    auto num_values = (result_size ? *result_size : result_size_);
    auto buffer_size = sizeof(T) * num_values;
    std::vector<T> buffer(num_values);
    cl_event event;
    auto prev_event = event_.get();
    auto err = clEnqueueReadBuffer(queue.get(), memory_.get(), CL_TRUE,
                                   0, buffer_size, buffer.data(),
                                   1, &prev_event, &event);
    if (err != CL_SUCCESS)
      return make_error(sec::runtime_error, get_opencl_error(err));
    return buffer;
  }

  /*
  void data(const actor&, atom) {
    // asynchronous function to read data from device
    // sends the results to dst with the marker atom
    throw std::runtime_error("Asynchronous mem_ref::data() not implemented");
  }
  */

  /*
  void move_to(device* dev) {
    // TODO: implement
  }
  */

  inline const mem_ptr& get() const {
    return memory_;
  }

  inline mem_ptr& get() {
    return memory_;
  }

  inline size_t size() const {
    return input_size_;
  }

  inline size_t result_size() const {
    return result_size_;
  }

  bool result_size(size_t size) {
    if (size > input_size_)
      return false;
    result_size_ = size;
    return true;
  }

  inline event_ptr event() const {
    return event_;
  }

  mem_ref()
    : input_size_{0},
      result_size_{0},
      device_{nullptr},
      event_{nullptr},
      memory_{nullptr} {
    // nop
  }
  mem_ref(std::vector<T> data, device* dev, mem_ptr memory, event_ptr event,
          optional<size_t> size = none)
    : input_size_{data.size()},
      result_size_{data.size()},
      device_{dev},
      event_{event},
      memory_{memory} {
    if (size)
      result_size_ = *size;
  }
  mem_ref(std::vector<T> data, device* dev, cl_mem memory, event_ptr event,
          optional<size_t> size = none)
    : input_size_{data.size()},
      result_size_{data.size()},
      device_{dev},
      event_{event},
      memory_{memory} {
    if (size)
      result_size_ = *size;
  }
  mem_ref(mem_ref&& other)
    : input_size_{std::move(other.input_size_)},
      result_size_{std::move(other.result_size_)},
      device_{std::move(other.device_)},
      event_{std::move(other.event_)},
      memory_{std::move(other.memory_)} {
    // nop
  }
  mem_ref& operator=(mem_ref&& other) {
    input_size_  = std::move(other.input_size_);
    result_size_ = std::move(other.result_size_);
    device_      = std::move(other.device_);
    event_       = std::move(other.event_);
    memory_      = std::move(other.memory_);
    return *this;
  }
  mem_ref(const mem_ref& other)
    : input_size_{other.input_size_},
      result_size_{other.result_size_},
      device_{other.device_},
      event_{other.event_},
      memory_{other.memory_} {
    // nop
  }
  mem_ref& operator=(const mem_ref& other) {
    if (&other == this)
      return *this;
    input_size_  = other.input_size_;
    result_size_ = other.result_size_;
    device_      = other.device_;
    event_       = other.event_;
    memory_      = other.memory_;
    return *this;
  }
  ~mem_ref() {
    // management ?
  }

private:
  void set_event(event_ptr event) {
    event_ = event;
  }

  size_t input_size_;
  size_t result_size_;
  buffer_type type_;
  device* device_;
  event_ptr event_; // TODO: use vector, regular cleanup of CL_COMPLETE events
  mem_ptr memory_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_MEM_REF_HPP
