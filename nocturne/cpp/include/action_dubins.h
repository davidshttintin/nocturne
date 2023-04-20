// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

namespace nocturne {

// Mirror class of Action, used to handle Extended Dubins Car Dynamics

class ActionDubins {
 public:
  ActionDubins() = default;
  ActionDubins(std::optional<float> acceleration, std::optional<float> steering_rate)
      : acceleration_(acceleration),
        steering_rate_(steering_rate) {}

  std::optional<float> acceleration() const { return acceleration_; }
  void set_acceleration(std::optional<float> acceleration) {
    acceleration_ = acceleration;
  }

  std::optional<float> steering_rate() const { return steering_rate_; }
  void set_steering_rate(std::optional<float> steering_rate) { steering_rate_ = steering_rate; }


 protected:
  std::optional<float> acceleration_ = std::nullopt;
  std::optional<float> steering_rate_ = std::nullopt;
};

}  // namespace nocturne
