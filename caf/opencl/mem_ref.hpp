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

/// A reference type for buffers on a OpenCL devive
template <class T>
class mem_ref : public ref_counted {
public:
  using value_type = T;

  friend class device;

  expected<std::vector<T>> data(optional<size_t> result_size = none) {
    if (!memory_)
      return make_error(sec::runtime_error, "No memory assigned.");
    if (is_local_)
      return make_error(sec::runtime_error, "Cannot read local memory");
    if (result_size && *result_size > num_elements_)
      return make_error(sec::runtime_error, "Buffer has less elements.");
    command_queue_ptr queue = device_->queue_;
    auto num_elements = (result_size ? *result_size : num_elements_);
    auto buffer_size = sizeof(T) * num_elements;
    std::vector<T> buffer(num_elements);
    cl_event event;
    std::vector<cl_event> prev_events;
    if (event_)
      prev_events.push_back(event_.get());
    auto err = clEnqueueReadBuffer(queue.get(), memory_.get(), CL_TRUE,
                                   0, buffer_size, buffer.data(),
                                   prev_events.size(), prev_events.data(),
                                   &event);
    if (err != CL_SUCCESS)
      return make_error(sec::runtime_error, get_opencl_error(err));
    return buffer;
  }

  void reset() {
    num_elements_ = 0;
    event_.reset();
    memory_.reset();
  }

  /*
  void data(const actor&, atom) {
    // asynchronous function to read data from device
    // sends the results to dst with the marker atom
    throw std::runtime_error("Asynchronous mem_ref::data() not implemented");
  }

  void move_to(device* dev) {
    // move buffer to different device
  }

  expected<mem_ref<T>> copy(mem_ref<T>) {
    // OpenCL function call
  }
  */

  inline const mem_ptr&  get()      const { return memory_;       }
  inline       mem_ptr&  get()            { return memory_;       }
  inline       size_t    size()     const { return num_elements_; }
  inline       event_ptr event()    const { return event_;        }
  inline       bool      is_local() const { return is_local_;     }

  mem_ref()
    : num_elements_{0},
      is_local_{false},
      device_{nullptr},
      event_{nullptr},
      memory_{nullptr} {
    // nop
  }
  mem_ref(size_t num_elements, bool is_local, device* dev, mem_ptr memory,
          event_ptr event)
    : num_elements_{num_elements},
      is_local_{is_local},
      device_{dev},
      event_{event},
      memory_{memory} {
    // nop
  }
  mem_ref(size_t num_elements, bool is_local, device* dev, cl_mem memory,
          event_ptr event)
    : num_elements_{num_elements},
      is_local_{is_local},
      device_{dev},
      event_{event},
      memory_{memory} {
    // nop
  }
  mem_ref(mem_ref&& other)
    : num_elements_{std::move(other.num_elements_)},
      is_local_{std::move(other.is_local_)},
      device_{std::move(other.device_)},
      event_{std::move(other.event_)},
      memory_{std::move(other.memory_)} {
    // nop
  }
  mem_ref& operator=(mem_ref&& other) {
    num_elements_ = std::move(other.num_elements_);
    is_local_ = std::move(other.is_local_);
    device_ = std::move(other.device_);
    event_ = std::move(other.event_);
    memory_ = std::move(other.memory_);
    return *this;
  }
  mem_ref(const mem_ref& other)
    : num_elements_{other.num_elements_},
      is_local_{other.is_local_},
      device_{other.device_},
      event_{other.event_},
      memory_{other.memory_} {
    // nop
  }
  mem_ref& operator=(const mem_ref& other) {
    if (&other == this)
      return *this;
    num_elements_  = other.num_elements_;
    is_local_ = other.is_local_;
    device_ = other.device_;
    event_ = other.event_;
    memory_ = other.memory_;
    return *this;
  }
  ~mem_ref() {
    // management ?
  }

  void swap(mem_ref<T>& other) {
    std::swap(num_elements_, other.num_elements_);
    std::swap(is_local_, other.is_local_);
    std::swap(type_, other.type_);
    std::swap(device_, other.device_);
    event_.swap(other.event_);
    memory_.swap(other.memory_);
    // auto tmp1 = other.event_;
    // other.event_ = event_;
    // event_ = tmp1;
    // auto tmp2 = other.memory_;
    // other.memory_ = memory_;
    // event_ = tmp2;
  }

private:
  void set_event(event_ptr event) {
    event_ = event;
  }

  size_t num_elements_;
  bool is_local_;
  buffer_type type_;
  device* device_;
  event_ptr event_; // TODO: use vector, regular cleanup of CL_COMPLETE events
  mem_ptr memory_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_MEM_REF_HPP
