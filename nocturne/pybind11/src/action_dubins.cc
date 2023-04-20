// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "action_dubins.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

namespace {

py::array_t<float> AsNumpyArray(const ActionDubins& action_dubins) {
  py::array_t<float> arr(2);
  float* arr_data = arr.mutable_data();
  arr_data[0] =
      action_dubins.acceleration().value_or(std::numeric_limits<float>::quiet_NaN());
  arr_data[1] =
      action_dubins.steering_rate().value_or(std::numeric_limits<float>::quiet_NaN());
  return arr;
}

ActionDubins FromNumpy(const py::array_t<float>& arr) {
  assert(arr.size() == 2);
  const float* arr_data = arr.data();
  std::optional<float> acceleration =
      std::isnan(arr_data[0]) ? std::nullopt
                              : std::make_optional<float>(arr_data[0]);
  std::optional<float> steering_rate = std::isnan(arr_data[1])
                                      ? std::nullopt
                                      : std::make_optional<float>(arr_data[1]);
  return ActionDubins(acceleration, steering_rate);
}

}  // namespace

void DefineActionDubins(py::module& m) {
  py::class_<ActionDubins>(m, "ActionDubins")
      .def(py::init<std::optional<float>, std::optional<float>>(),
           py::arg("acceleration") = py::none(),
           py::arg("steering_rate") = py::none())
      .def("__repr__",
           [](const ActionDubins& act) {
             const std::string acceleration_str =
                 act.acceleration().has_value()
                     ? std::to_string(act.acceleration().value())
                     : "None";
             const std::string steering_rate_str =
                 act.steering_rate().has_value()
                     ? std::to_string(act.steering_rate().value())
                     : "None";
             return "{acceleration: " + acceleration_str +
                    ", steering_rate: " + steering_rate_str + "}";
           })
      .def_property("acceleration", &ActionDubins::acceleration,
                    &ActionDubins::set_acceleration)
      .def_property("steering_rate", &ActionDubins::steering_rate, &ActionDubins::set_steering_rate)
      .def("numpy", &AsNumpyArray)
      .def_static("from_numpy", &FromNumpy)
      .def(py::pickle(
          [](const ActionDubins& act) { return AsNumpyArray(act); },
          [](const py::array_t<float>& arr) { return FromNumpy(arr); }));
}

}  // namespace nocturne
