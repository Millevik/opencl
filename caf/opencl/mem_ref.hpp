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

#include <ios>
#include <vector>

#include "caf/sec.hpp"
#include "caf/optional.hpp"
#include "caf/ref_counted.hpp"

#include "caf/opencl/smart_ptr.hpp"

namespace caf {
namespace opencl {

struct msg_adding_event {
  msg_adding_event(cl_event event) : event_(event) {
    // nop
  }
  template <class T, class... Ts>
  message operator()(T& x, Ts&... xs) {
    return make_message(add_event(std::move(x)), add_event(std::move(xs))...);
  }
  template <class... Ts>
  message operator()(std::tuple<Ts...>& values) {
    return apply_args(*this, detail::get_indices(values), values);
  }
  template <class T>
  mem_ref<T> add_event(mem_ref<T> ref) {
    ref.set_event(event_);
    return std::move(ref);
  }
  cl_event event_;
};


/// A reference type for buffers on a OpenCL devive. Access is not thread safe.
/// Hence, a mem_ref should only be passed to actors sequentially.
template <class T>
class mem_ref {
public:
  using value_type = T;

  friend struct msg_adding_event;
  template <class... Ts>
  friend class actor_facade_phase;
  template <class... Ts>
  friend class opencl_actor;

  expected<std::vector<T>> data(optional<size_t> result_size = none) {
    switch (location_) {
      case placement::uninitialized:
        return make_error(sec::runtime_error,
                          "Cannot read uninitialized memory.");
      case placement::local_mem:
        return make_error(sec::runtime_error, "Cannot read local memory.");
      case placement::private_mem:
        return make_error(sec::runtime_error, "Cannot read private memory.");
      case placement::global_mem: {
        if (0 != (access_ & CL_MEM_HOST_NO_ACCESS))
          return make_error(sec::runtime_error, "No memory access.");
        if (!memory_)
          return make_error(sec::runtime_error, "No memory assigned.");
        if (result_size && *result_size > num_elements_)
          return make_error(sec::runtime_error, "Buffer has less elements.");
        auto num_elements = (result_size ? *result_size : num_elements_);
        auto buffer_size = sizeof(T) * num_elements;
        std::vector<T> buffer(num_elements);
        cl_event event;
        std::vector<cl_event> prev_events;
        if (event_ != nullptr)
          prev_events.push_back(event_);
        auto err = clEnqueueReadBuffer(queue_.get(), memory_.get(), CL_TRUE,
                                       0, buffer_size, buffer.data(),
                                       static_cast<cl_uint>(prev_events.size()),
                                       prev_events.data(), &event);
        if (err != CL_SUCCESS)
          return make_error(sec::runtime_error, get_opencl_error(err));
        // decrements the previous event we used for waiting above
        set_event(event, false);
        return buffer;
      }
    }
    return make_error(sec::runtime_error, "Unknown memory location.");
  }

  /*
  void insert(const vector<T>& data, optional<size_t> from = none,
              optional<size_t> to = none) {
    
  }
  */

  void reset() {
    num_elements_ = 0;
    location_ = placement::uninitialized;
    access_ = CL_MEM_HOST_NO_ACCESS;
    memory_.reset();
    value_ = none;
    access_ = 0;
    clear_event();
  }

  /*
  
  /// wait for the event if available
  bool sync() {

  }

  void data(const actor&, atom) {
    // asynchronous function to read data from device
    // sends the results to dst with the marker atom
    throw std::runtime_error("Asynchronous mem_ref::data() not implemented");
  }
  */

  expected<mem_ref<T>> copy() {
    switch (location_) {
      case placement::uninitialized:
      case placement::local_mem:
      case placement::private_mem: {
        mem_ref<T> new_ref = *this;
        return new_ref;
      }
      case placement::global_mem: {
        if (!memory_)
          return make_error(sec::runtime_error, "No memory assigned.");
        auto buffer_size = sizeof(T) * num_elements_;
        cl_event event;
        cl_mem buffer;
        std::vector<cl_event> prev_events;
        if (event_ != nullptr)
          prev_events.push_back(event_);
        auto err = clEnqueueCopyBuffer(queue_.get(), memory_.get(), buffer,
                                       0, 0, // no offset for now
                                       buffer_size, prev_events.size(),
                                       prev_events.data(), &event);
        if (err != CL_SUCCESS)
          return make_error(sec::runtime_error, get_opencl_error(err));
        // decrements the previous event we used for waiting above
        set_event(event, false);
        return mem_ref<T>(num_elements_, location_, queue_, std::move(buffer),
                          access_, event, true);
      }
    }
  }

  bool synchronize() {
    if (event_ == nullptr)
      return false;
    auto err = clWaitForEvents(1, &event_);
    if (err != CL_SUCCESS)
      return false;
    return true;
  }

  inline const mem_ptr& get() const {
    return memory_;
  }

  inline mem_ptr& get() {
    return memory_;
  }

  inline size_t size() const {
    return num_elements_;
  }

  inline placement location() const {
    return location_;
  }

  /// Only available for private arguments, return none otherwise
  inline optional<T> value() const {
    return value_;
  }

  mem_ref()
    : num_elements_{0},
      location_{placement::uninitialized},
      access_{CL_MEM_HOST_NO_ACCESS},
      queue_{nullptr},
      event_{nullptr},
      memory_{nullptr},
      value_{none} {
    // nop
  }

  mem_ref(size_t num_elements, placement location, command_queue_ptr queue,
          mem_ptr memory, cl_mem_flags access, cl_event event,
          bool inc_event_ref = true, optional<T> value = none)
    : num_elements_{num_elements},
      location_{location},
      access_{access},
      queue_{queue},
      event_{event},
      memory_{memory},
      value_{std::move(value)} {
    if (inc_event_ref)
      inc_event();
  }

  /// mem_ref assumes that the event passed to it already has an incremented
  /// reference count on the event.
  mem_ref(size_t num_elements, placement location, command_queue_ptr queue,
          cl_mem memory, cl_mem_flags access, cl_event event,
          bool inc_event_ref = true, optional<T> value = none)
    : num_elements_{num_elements},
      location_{location},
      access_{access},
      queue_{queue},
      event_{event},
      memory_{memory},
      value_{std::move(value)} {
    if (inc_event_ref)
      inc_event();
  }

  mem_ref(mem_ref&& other)
    : num_elements_{other.num_elements_},
      location_{other.location_},
      access_{other.access_},
      queue_{other.queue_},
      event_{other.event_},
      memory_{other.memory_},
      value_{other.value_} {
    // null other's ref to keep the ref count
    other.event_ = nullptr;
    other.reset();
  }

  mem_ref& operator=(mem_ref<T>&& other) {
    num_elements_ = other.num_elements_;
    location_ = other.location_;
    access_ = other.access_;
    queue_ = other.queue_;
    memory_ = other.memory_;
    value_ = other.value_;
    // decrement our previous event
    dec_event();
    // take event and null other's ref to keep the ref count
    event_ = other.event_;
    other.event_ = nullptr;
    other.reset();
    return *this;
  }

  mem_ref(const mem_ref& other) {
    num_elements_ = other.num_elements_;
    location_ = other.location_;
    access_ = other.access_;
    queue_ = other.queue_;
    memory_ = other.memory_;
    value_ = other.value_;
    event_ = other.event_;
    inc_event();
  }

  mem_ref& operator=(const mem_ref& other) {
    num_elements_ = other.num_elements_;
    location_ = other.location_;
    access_ = other.access_;
    queue_ = other.queue_;
    memory_ = other.memory_;
    value_ = other.value_;
    event_ = other.event_;
    inc_event();
    return *this;
  }

  ~mem_ref() {
    if (event_ != nullptr)
      clReleaseEvent(event_);
  };

  void swap(mem_ref<T>& other) {
    std::swap(num_elements_, other.num_elements_);
    std::swap(location_, other.location_);
    std::swap(access_, other.access_);
    std::swap(queue_, other.queue_);
    std::swap(event_, other.event_);
    memory_.swap(other.memory_);
    std::swap(value_, other.value_);
  }

private:
  inline void set_event(cl_event e, bool increment_reference = true) {
    dec_event();
    event_ = e;
    if (increment_reference)
      inc_event();
  }

  inline cl_event event(bool increment_reference = true) {
    if (increment_reference)
      inc_event();
    return event_;
  }

  inline cl_event take_event() {
    auto event = event_;
    event_ = nullptr;
    return event;
  }

  inline void clear_event() {
    dec_event();
    event_ = nullptr;
  }

  inline void inc_event() {
    if (event_ != nullptr)
      clRetainEvent(event_);
  }

  inline void dec_event() {
    if (event_ != nullptr)
      clReleaseEvent(event_);
  }

  size_t num_elements_;
  placement location_;
  cl_mem_flags access_;
  command_queue_ptr queue_;
  cl_event event_;
  mem_ptr memory_;
  optional<T> value_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_MEM_REF_HPP
