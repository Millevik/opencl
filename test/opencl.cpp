#define CAF_SUITE opencl
#include "caf/test/unit_test.hpp"

#include <vector>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "caf/all.hpp"
#include "caf/system_messages.hpp"

#include "caf/opencl/all.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

using caf::detail::limited_vector;

// required to allow sending mem_ref<int> in messages
namespace caf {
  template <>
  struct allowed_unsafe_message_type<mem_ref<int>> : std::true_type {};
}

namespace {

using ivec = vector<int>;
using dims = opencl::dim_vec;

constexpr size_t matrix_size = 4;
constexpr size_t array_size = 32;
constexpr size_t problem_size = 1024;

constexpr const char* kernel_name = "matrix_square";
constexpr const char* kernel_name_compiler_flag = "compiler_flag";
constexpr const char* kernel_name_reduce = "reduce";
constexpr const char* kernel_name_const = "const_mod";
constexpr const char* kernel_name_inout = "times_two";
constexpr const char* kernel_name_buffer = "times_two";

constexpr const char* compiler_flag = "-D CAF_OPENCL_TEST_FLAG";

constexpr const char* kernel_source = R"__(
  __kernel void matrix_square(__global int* matrix,
                              __global int* output) {
    size_t size = get_global_size(0); // == get_global_size_(1);
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    int result = 0;
    for (size_t idx = 0; idx < size; ++idx) {
      result += matrix[idx + y * size] * matrix[x + idx * size];
    }
    output[x + y * size] = result;
  }
)__";

constexpr const char* kernel_source_error = R"__(
  __kernel void missing(__global int*) {
    size_t semicolon_missing
  }
)__";

constexpr const char* kernel_source_compiler_flag = R"__(
  __kernel void compiler_flag(__global int* input,
                              __global int* output) {
    size_t x = get_global_id(0);
#   ifdef CAF_OPENCL_TEST_FLAG
    output[x] = input[x];
#   else
    output[x] = 0;
#   endif
  }
)__";

// http://developer.amd.com/resources/documentation-articles/articles-whitepapers/
// opencl-optimization-case-study-simple-reductions
constexpr const char* kernel_source_reduce = R"__(
  __kernel void reduce(__global int* buffer,
                       __global int* result) {
    __local int scratch[512];
    int local_index = get_local_id(0);
    scratch[local_index] = buffer[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
      if (local_index < offset) {
        int other = scratch[local_index + offset];
        int mine = scratch[local_index];
        scratch[local_index] = (mine < other) ? mine : other;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
      result[get_group_id(0)] = scratch[0];
    }
  }
)__";

constexpr const char* kernel_source_const = R"__(
  __kernel void const_mod(__constant int* input,
                          __global int* output) {
    size_t idx = get_global_id(0);
    output[idx] = input[0];
  }
)__";

constexpr const char* kernel_source_inout = R"__(
  __kernel void times_two(__global int* values) {
    size_t idx = get_global_id(0);
    values[idx] = values[idx] * 2;
  }
)__";

constexpr const char* kernel_source_buffer = R"__(
  __kernel void times_two(__global int* values,
                          __global int* buf) {
    size_t idx = get_global_id(0);
    buf[idx] = values[idx];
    buf[idx] += values[idx];
    values[idx] = buf[idx];
  }
)__";

} // namespace <anonymous>

template<size_t Size>
class square_matrix {
public:
  using value_type = ivec::value_type;
  static constexpr size_t num_elements = Size * Size;

  template <class Inspector>
  friend typename Inspector::result_type inspect(Inspector& f,
                                                 square_matrix& x) {
    return f(meta::type_name("square_matrix"), x.data_);
  }

  square_matrix(square_matrix&&) = default;
  square_matrix(const square_matrix&) = default;
  square_matrix& operator=(square_matrix&&) = default;
  square_matrix& operator=(const square_matrix&) = default;

  square_matrix() : data_(num_elements) {
    // nop
  }

  explicit square_matrix(ivec d) : data_(move(d)) {
    assert(data_.size() == num_elements);
  }

  float& operator()(size_t column, size_t row) {
    return data_[column + row * Size];
  }

  const float& operator()(size_t column, size_t row) const {
    return data_[column + row * Size];
  }

  using const_iterator = typename ivec::const_iterator;

  const_iterator begin() const {
    return data_.begin();
  }

  const_iterator end() const {
    return data_.end();
  }

  ivec& data() {
    return data_;
  }

  const ivec& data() const {
    return data_;
  }

  void data(ivec new_data) {
    data_ = move(new_data);
  }

private:
  ivec data_;
};


template <class T>
vector<T> make_iota_vector(size_t num_elements) {
  vector<T> result;
  result.resize(num_elements);
  iota(result.begin(), result.end(), T{0});
  return result;
}

template <size_t Size>
square_matrix<Size> make_iota_matrix() {
  square_matrix<Size> result;
  iota(result.data().begin(), result.data().end(), 0);
  return result;
}

template<size_t Size>
bool operator==(const square_matrix<Size>& lhs,
                const square_matrix<Size>& rhs) {
  return lhs.data() == rhs.data();
}

template<size_t Size>
bool operator!=(const square_matrix<Size>& lhs,
                const square_matrix<Size>& rhs) {
  return !(lhs == rhs);
}

using matrix_type = square_matrix<matrix_size>;

template <class T>
void check_vector_results(const string& description,
                          const vector<T>& expected,
                          const vector<T>& result) {
  auto cond = (expected == result);
  CAF_CHECK(cond);
  if (!cond) {
    CAF_ERROR(description << " failed.");
    cout << "Expected: " << endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      cout << " " << expected[i];
    }
    cout << endl << "Received: " << endl;
    for (size_t i = 0; i < result.size(); ++i) {
      cout << " " << result[i];
    }
    cout << endl;
  }
}

void test_opencl(actor_system& sys) {
  auto& mngr = sys.opencl_manager();
  auto opt = mngr.get_device_if([](const device&){ return true; });
  CAF_REQUIRE(opt);
  auto dev = *opt;
  scoped_actor self{sys};
  const ivec expected1{ 56,  62,  68,  74,
                       152, 174, 196, 218,
                       248, 286, 324, 362,
                       344, 398, 452, 506};
  auto w1 = mngr.spawn(mngr.create_program(kernel_source, "", dev), kernel_name,
                       opencl::spawn_config{dims{matrix_size, matrix_size}},
                       opencl::in<int*>{}, opencl::out<int*>{});
  self->send(w1, make_iota_vector<int>(matrix_size * matrix_size));
  self->receive (
    [&](const ivec& result) {
      check_vector_results("Simple matrix multiplication using vectors"
                           "(kernel wrapped in program)",
                           expected1, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  opencl::spawn_config cfg2{dims{matrix_size, matrix_size}};
  auto w2 = mngr.spawn(kernel_source, kernel_name, cfg2,
                       opencl::in<int*>{}, opencl::out<int*>{});
  self->send(w2, make_iota_vector<int>(matrix_size * matrix_size));
  self->receive (
    [&](const ivec& result) {
      check_vector_results("Simple matrix multiplication using vectors",
                           expected1, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  const matrix_type expected2(move(expected1));
  auto map_arg = [](message& msg) -> optional<message> {
    return msg.apply(
      [](matrix_type& mx) {
        return make_message(move(mx.data()));
      }
    );
  };
  auto map_res = [=](ivec result) -> message {
    return make_message(matrix_type{move(result)});
  };
  opencl::spawn_config cfg3{dims{matrix_size, matrix_size}};
  auto w3 = mngr.spawn(mngr.create_program(kernel_source), kernel_name, cfg3,
                       map_arg, map_res,
                       opencl::in<int*>{}, opencl::out<int*>{});
  self->send(w3, make_iota_matrix<matrix_size>());
  self->receive (
    [&](const matrix_type& result) {
      check_vector_results("Matrix multiplication with user defined type "
                           "(kernel wrapped in program)",
                           expected2.data(), result.data());
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  opencl::spawn_config cfg4{dims{matrix_size, matrix_size}};
  auto w4 = mngr.spawn(kernel_source, kernel_name, cfg4,
                       map_arg, map_res,
                       opencl::in<int*>{}, opencl::out<int*>{});
  self->send(w4, make_iota_matrix<matrix_size>());
  self->receive (
    [&](const matrix_type& result) {
      check_vector_results("Matrix multiplication with user defined type",
                           expected2.data(), result.data());
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  CAF_MESSAGE("Expecting exception (compiling invalid kernel, "
              "semicolon is missing).");
  try {
    /* auto create_error = */ mngr.create_program(kernel_source_error);
  }
  catch (const exception& exc) {
    auto cond = (strcmp("clBuildProgram: CL_BUILD_PROGRAM_FAILURE",
                        exc.what()) == 0);
      CAF_CHECK(cond);
      if (!cond) {
        CAF_ERROR("Wrong exception cought for program build failure.");
      }
  }
  // test for opencl compiler flags
  auto prog5 = mngr.create_program(kernel_source_compiler_flag, compiler_flag);
  opencl::spawn_config cfg5{dims{array_size}};
  auto w5 = mngr.spawn(prog5, kernel_name_compiler_flag, cfg5,
                     opencl::in<int*>{}, opencl::out<int*>{});
  self->send(w5, make_iota_vector<int>(array_size));
  auto expected3 = make_iota_vector<int>(array_size);
  self->receive (
    [&](const ivec& result) {
      check_vector_results("Passing compiler flags", expected3, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );

  auto dev6 = mngr.get_device_if([](const device& d) {
    return d.get_device_type() != cpu;
  });
  if (dev6) {
    // test for manuel return size selection (max workgroup size 1d)
    auto max_wg_size = min(dev6->get_max_work_item_sizes()[0], size_t{512});
    auto reduce_buffer_size = static_cast<size_t>(max_wg_size) * 8;
    auto reduce_local_size  = static_cast<size_t>(max_wg_size);
    auto reduce_work_groups = reduce_buffer_size / reduce_local_size;
    auto reduce_global_size = reduce_buffer_size;
    auto reduce_result_size = reduce_work_groups;
    ivec arr6(reduce_buffer_size);
    int n = static_cast<int>(arr6.capacity());
    generate(arr6.begin(), arr6.end(), [&]{ return --n; });
    opencl::spawn_config cfg6{dims{reduce_global_size},
                              dims{                  }, // no offset
                              dims{reduce_local_size}};
    auto get_result_size_6 = [reduce_result_size](const ivec&) {
      return reduce_result_size;
    };
    auto w6 = mngr.spawn(mngr.create_program(kernel_source_reduce, "", *dev6),
                         kernel_name_reduce, cfg6, opencl::in<ivec>{},
                         opencl::out<ivec>{get_result_size_6});
    self->send(w6, move(arr6));
    auto wg_size_as_int = static_cast<int>(max_wg_size);
    ivec expected4{wg_size_as_int * 7, wg_size_as_int * 6, wg_size_as_int * 5,
                   wg_size_as_int * 4, wg_size_as_int * 3, wg_size_as_int * 2,
                   wg_size_as_int    ,               0};
    self->receive(
      [&](const ivec& result) {
        check_vector_results("Passing size for the output", expected4, result);
      },
      others >> [&](message_view& x) -> result<message> {
        CAF_ERROR("unexpected message" << x.content().stringify());
        return sec::unexpected_message;
      }
    );
  }
  // calculator function for getting the size of the output
  auto get_result_size_7 = [=](const ivec&) {
    return problem_size;
  };
  // constant memory arguments
  const ivec arr7{problem_size};
  auto w7 = mngr.spawn(kernel_source_const, kernel_name_const,
                       opencl::spawn_config{dims{problem_size}},
                       opencl::in<ivec>{},
                       opencl::out<ivec>{get_result_size_7});
  self->send(w7, move(arr7));
  ivec expected5(problem_size);
  fill(begin(expected5), end(expected5), static_cast<int>(problem_size));
  self->receive(
    [&](const ivec& result) {
      check_vector_results("Using const input argument", expected5, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  // test in_out argument type
  ivec input9 = make_iota_vector<int>(problem_size);
  ivec expected9{input9};
  for_each(begin(expected9), end(expected9), [](int& val){ val *= 2; });
  auto w9 = mngr.spawn(kernel_source_inout, kernel_name_inout,
                       spawn_config{dims{problem_size}},
                       opencl::in_out<ivec>{});
  self->send(w9, move(input9));
  self->receive(
    [&](const ivec& result) {
      check_vector_results("Testing in_out arugment", expected9, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  // test buffer argument type
  ivec input10 = make_iota_vector<int>(problem_size);
  ivec expected10{input10};
  for_each(begin(expected10), end(expected10), [](int& val){ val *= 2; });
  auto get_result_size_10 = [=](const ivec& input) {
    return input.size();
  };
  auto w10 = mngr.spawn(kernel_source_buffer, kernel_name_buffer,
                        spawn_config{dims{problem_size}},
                        opencl::in_out<ivec>{},
                        opencl::scratch<ivec>{get_result_size_10});
  self->send(w10, move(input10));
  self->receive(
    [&](const ivec& result) {
      check_vector_results("Testing buffer arugment", expected10, result);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
}

void test_phases(actor_system& sys) {
  auto& mngr = sys.opencl_manager();
  auto opt = mngr.get_device(0);
  CAF_REQUIRE(opt);
  auto dev = *opt;
  scoped_actor self{sys};
  ivec input = make_iota_vector<int>(problem_size);
  ivec expected{input};
  for_each(begin(expected), end(expected), [](int& val) { val *= 2; });
  auto prog   = mngr.create_program(kernel_source_inout, "", dev);
  auto conf   = spawn_config{dims{input.size()}};
  auto worker = mngr.spawn_phase<int*>(prog, kernel_name_inout, conf);
  auto buf    = dev.copy_to_device(buffer_type::input_output, input);
  CAF_CHECK(buf.size(), input.size());
  self->send(worker, buf);
  self->receive(
    [&](mem_ref<int>& ref) {
      auto res = ref.data();
      CAF_CHECK(res);
      check_vector_results("Testing phase one", expected, *res);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
  for_each(begin(expected), end(expected), [](int& val) { val *= 2; });
  self->send(worker, buf);
  self->receive(
    [&](mem_ref<int>& ref) {
      auto res = ref.data();
      CAF_CHECK(res);
      check_vector_results("Testing phase one", expected, *res);
    },
    others >> [&](message_view& x) -> result<message> {
      CAF_ERROR("unexpected message" << x.content().stringify());
      return sec::unexpected_message;
    }
  );
}

CAF_TEST(opencl_basics) {
  actor_system_config cfg;
  cfg.load<opencl::manager>()
    .add_message_type<ivec>("int_vector")
    .add_message_type<matrix_type>("square_matrix");
  actor_system system{cfg};
  test_opencl(system);
  system.await_all_actors_done();
}

CAF_TEST(opencl_mem_refs) {
  actor_system_config cfg;
  cfg.load<opencl::manager>();
  actor_system system{cfg};
  auto& mngr = system.opencl_manager();
  auto opt = mngr.get_device(0);
  CAF_REQUIRE(opt);
  auto dev = *opt;
  vector<uint32_t> input{1, 2, 3, 4};
  auto buf_1 = dev.copy_to_device(buffer_type::input_output, input);
  CAF_CHECK_EQUAL(buf_1.size(), input.size());
  CAF_CHECK_EQUAL(buf_1.result_size(), input.size());
  auto res_1 = buf_1.data();
  CAF_CHECK(res_1);
  CAF_CHECK_EQUAL(res_1->size(), input.size());
  check_vector_results("Testing mem_ref", input, *res_1);
  auto res_2 = buf_1.data(2);
  CAF_CHECK(res_2);
  CAF_CHECK_EQUAL(res_2->size(), 2ul);
  CAF_CHECK_EQUAL((*res_2)[0], input[0]);
  CAF_CHECK_EQUAL((*res_2)[1], input[1]);
  vector<uint32_t> new_input{1,2,3,4,5};
  buf_1 = dev.copy_to_device(buffer_type::input_output, new_input);
  auto res_3 = buf_1.data();
  CAF_CHECK_EQUAL(buf_1.size(), new_input.size());
  CAF_CHECK_EQUAL(buf_1.result_size(), new_input.size());
  buf_1.reset();
  auto res_4 = buf_1.data();
  CAF_CHECK(!res_4);
}

CAF_TEST(opencl_phase_facade) {
  actor_system_config cfg;
  cfg.load<opencl::manager>()
    .add_message_type<ivec>("int_vector");
  actor_system system{cfg};
  test_phases(system);
  system.await_all_actors_done();
}

