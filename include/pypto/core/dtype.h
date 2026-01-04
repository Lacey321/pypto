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

/**
 * @file dtype.h
 * @brief Data type definitions for PyPTO tensors and operations
 *
 * This file defines the DataType enum which represents all supported numeric types
 * in the PyPTO framework, including integers, unsigned integers, floating point,
 * bfloat16, and hybrid float formats.
 */

#ifndef PYPTO_CORE_DTYPE_H_
#define PYPTO_CORE_DTYPE_H_

#include <cstdint>
#include <string>

namespace pypto {

/**
 * @brief Enumeration of all supported data types in PyPTO
 *
 * This enum defines all numeric data types supported by PyPTO tensors and operations.
 * It includes:
 * - Signed integers: INT4, INT8, INT16, INT32, INT64
 * - Unsigned integers: UINT4, UINT8, UINT16, UINT32, UINT64
 * - Floating point: FP4, FP8, FP16, FP32
 * - Brain floating point: BF16
 * - Hybrid float formats: HF4, HF8
 * - Boolean: BOOL
 */
enum class DataType : uint8_t {
  // Boolean type
  BOOL = 0,  // Boolean (true/false)

  // Signed integer types
  INT4 = 1,   // 4-bit signed integer
  INT8 = 2,   // 8-bit signed integer
  INT16 = 3,  // 16-bit signed integer
  INT32 = 4,  // 32-bit signed integer
  INT64 = 5,  // 64-bit signed integer

  // Unsigned integer types
  UINT4 = 6,    // 4-bit unsigned integer
  UINT8 = 7,    // 8-bit unsigned integer
  UINT16 = 8,   // 16-bit unsigned integer
  UINT32 = 9,   // 32-bit unsigned integer
  UINT64 = 10,  // 64-bit unsigned integer

  // Floating point types
  FP4 = 11,   // 4-bit floating point
  FP8 = 12,   // 8-bit floating point
  FP16 = 13,  // 16-bit floating point (IEEE 754 half precision)
  FP32 = 14,  // 32-bit floating point (IEEE 754 single precision)
  BF16 = 15,  // 16-bit brain floating point

  // Hisilicon float types
  HF4 = 16,  // 4-bit Hisilicon float
  HF8 = 17,  // 8-bit Hisilicon float
};

/**
 * @brief Get the size in bits of a given data type
 *
 * Returns the storage size in bits for each data type. This accurately
 * represents sub-byte types like INT4, UINT4, FP4, and HF4.
 *
 * @param dtype The data type to query
 * @return Size in bits
 */
inline size_t GetDataTypeBit(DataType dtype) {
  switch (dtype) {
    case DataType::INT4:
    case DataType::UINT4:
    case DataType::FP4:
    case DataType::HF4:
      return 4;
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::FP8:
    case DataType::HF8:
    case DataType::BOOL:
      return 8;
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FP16:
    case DataType::BF16:
      return 16;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FP32:
      return 32;
    case DataType::INT64:
    case DataType::UINT64:
      return 64;
    default:
      return 0;
  }
}

/**
 * @brief Get a human-readable string name for a data type
 *
 * @param dtype The data type to convert to string
 * @return String representation of the data type
 */
inline std::string DataTypeToString(DataType dtype) {
  switch (dtype) {
    case DataType::INT4:
      return "int4";
    case DataType::INT8:
      return "int8";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::UINT4:
      return "uint4";
    case DataType::FP4:
      return "fp4";
    case DataType::FP8:
      return "fp8";
    case DataType::FP16:
      return "fp16";
    case DataType::FP32:
      return "fp32";
    case DataType::BF16:
      return "bfloat16";
    case DataType::HF4:
      return "hf4";
    case DataType::HF8:
      return "hf8";
    case DataType::UINT8:
      return "uint8";
    case DataType::UINT16:
      return "uint16";
    case DataType::UINT32:
      return "uint32";
    case DataType::UINT64:
      return "uint64";
    case DataType::BOOL:
      return "bool";
    default:
      return "unknown";
  }
}

/**
 * @brief Check if a data type is a floating point type
 *
 * @param dtype The data type to check
 * @return true if dtype is FP4, FP8, FP16, FP32, BF16, HF4, or HF8
 */
inline bool IsFloat(DataType dtype) {
  return dtype == DataType::FP4 || dtype == DataType::FP8 || dtype == DataType::FP16 ||
         dtype == DataType::FP32 || dtype == DataType::BF16 || dtype == DataType::HF4 ||
         dtype == DataType::HF8;
}

/**
 * @brief Check if a data type is a signed integer type
 *
 * @param dtype The data type to check
 * @return true if dtype is INT4, INT8, INT16, INT32, or INT64
 */
inline bool IsSignedInt(DataType dtype) {
  return dtype == DataType::INT4 || dtype == DataType::INT8 || dtype == DataType::INT16 ||
         dtype == DataType::INT32 || dtype == DataType::INT64;
}

/**
 * @brief Check if a data type is an unsigned integer type
 *
 * @param dtype The data type to check
 * @return true if dtype is UINT4, UINT8, UINT16, UINT32, or UINT64
 */
inline bool IsUnsignedInt(DataType dtype) {
  return dtype == DataType::UINT4 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
         dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

/**
 * @brief Check if a data type is any integer type (signed or unsigned)
 *
 * @param dtype The data type to check
 * @return true if dtype is any integer type
 */
inline bool IsInt(DataType dtype) { return IsSignedInt(dtype) || IsUnsignedInt(dtype); }

}  // namespace pypto

#endif  // PYPTO_CORE_DTYPE_H_
