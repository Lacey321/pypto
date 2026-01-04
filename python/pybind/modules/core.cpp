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

#include <pybind11/pybind11.h>

#include "../bindings.h"
#include "pypto/core/dtype.h"

namespace py = pybind11;

namespace pypto {
namespace python {

void BindCore(py::module_& m) {
  // Bind DataType enum
  py::enum_<DataType>(m, "DataType", "Enumeration of all supported data types in PyPTO")
      // Signed integer types
      .value("INT4", DataType::INT4, "4-bit signed integer")
      .value("INT8", DataType::INT8, "8-bit signed integer")
      .value("INT16", DataType::INT16, "16-bit signed integer")
      .value("INT32", DataType::INT32, "32-bit signed integer")
      .value("INT64", DataType::INT64, "64-bit signed integer")
      // Unsigned integer types
      .value("UINT4", DataType::UINT4, "4-bit unsigned integer")
      .value("UINT8", DataType::UINT8, "8-bit unsigned integer")
      .value("UINT16", DataType::UINT16, "16-bit unsigned integer")
      .value("UINT32", DataType::UINT32, "32-bit unsigned integer")
      .value("UINT64", DataType::UINT64, "64-bit unsigned integer")
      // Floating point types
      .value("FP4", DataType::FP4, "4-bit floating point")
      .value("FP8", DataType::FP8, "8-bit floating point")
      .value("FP16", DataType::FP16, "16-bit floating point (IEEE 754 half precision)")
      .value("FP32", DataType::FP32, "32-bit floating point (IEEE 754 single precision)")
      .value("BF16", DataType::BF16, "16-bit brain floating point")
      // Hybrid float types
      .value("HF4", DataType::HF4, "4-bit hybrid float")
      .value("HF8", DataType::HF8, "8-bit hybrid float")
      // Boolean type
      .value("BOOL", DataType::BOOL, "Boolean (true/false)")
      .export_values();  // Export values to module scope for convenience

  // Bind helper functions
  m.def(
      "get_dtype_bit", &GetDataTypeBit, py::arg("dtype"),
      "Get the size in bits of a data type. Returns the actual bit size for sub-byte types (e.g., 4 bits for "
      "INT4).");

  m.def("dtype_to_string", &DataTypeToString, py::arg("dtype"),
        "Get a human-readable string name for a data type.");

  m.def("is_float", &IsFloat, py::arg("dtype"),
        "Check if a data type is a floating point type (FP4, FP8, FP16, FP32, BF16, HF4, HF8).");

  m.def("is_signed_int", &IsSignedInt, py::arg("dtype"),
        "Check if a data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).");

  m.def("is_unsigned_int", &IsUnsignedInt, py::arg("dtype"),
        "Check if a data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).");

  m.def("is_int", &IsInt, py::arg("dtype"), "Check if a data type is any integer type (signed or unsigned).");
}

}  // namespace python
}  // namespace pypto
