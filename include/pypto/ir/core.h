/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_CORE_H_
#define PYPTO_IR_CORE_H_

#include <memory>
#include <string>
#include <utility>

namespace pypto {
namespace ir {

/**
 * @brief Source location information for IR nodes
 *
 * Tracks the exact position in source code where an IR node originated.
 * Immutable value type - all fields are const.
 */
class Span {
 public:
  const std::string filename_;  ///< Source filename
  const int begin_line_;        ///< Beginning line number (1-indexed)
  const int begin_column_;      ///< Beginning column number (1-indexed)
  const int end_line_;          ///< Ending line number (1-indexed), -1 means unknown
  const int end_column_;        ///< Ending column number (1-indexed), -1 means unknown

  /**
   * @brief Construct a source span
   *
   * @param file Source filename
   * @param bl Begin line (1-indexed)
   * @param bc Begin column (1-indexed)
   * @param el End line (1-indexed)
   * @param ec End column (1-indexed)
   */
  Span(std::string file, int begin_line, int begin_column, int end_line = -1, int end_column = -1);

  /**
   * @brief Convert span to string representation
   *
   * @return String in format "filename:begin_line:begin_column"
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief Check if the span is valid (has valid line/column numbers)
   *
   * @return true if all line/column numbers are positive
   */
  [[nodiscard]] bool is_valid() const;

  /**
   * @brief Create an unknown/invalid span
   *
   * @return Span with empty filename and invalid coordinates
   */
  static Span unknown();
};

/**
 * @brief Base class for all IR nodes
 *
 * Abstract base providing common functionality for all IR nodes.
 * All IR nodes are immutable - once constructed, they cannot be modified.
 */
class IRNode {
 public:
  explicit IRNode(Span s) : span(std::move(s)) {}
  virtual ~IRNode() = default;

  // Disable copying and moving to enforce immutability
  IRNode(IRNode&&) = delete;
  IRNode& operator=(IRNode&&) = delete;

  Span span;  // Source location
};
using IRNodePtr = std::shared_ptr<const IRNode>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_CORE_H_
