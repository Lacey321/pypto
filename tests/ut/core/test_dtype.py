# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the DataType enum and related utility functions."""

import pypto
import pytest
from pypto import (
    DT_BF16,
    DT_BOOL,
    DT_FP4,
    DT_FP8,
    DT_FP16,
    DT_FP32,
    DT_HF4,
    DT_HF8,
    DT_INT4,
    DT_INT8,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_UINT4,
    DT_UINT8,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
    DataType,
    dtype_to_string,
    get_dtype_bit,
    is_float,
    is_int,
    is_signed_int,
    is_unsigned_int,
)


class TestDataTypeEnum:
    """Test DataType enumeration values and access patterns."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        # Signed integers
        assert hasattr(DataType, "INT4")
        assert hasattr(DataType, "INT8")
        assert hasattr(DataType, "INT16")
        assert hasattr(DataType, "INT32")
        assert hasattr(DataType, "INT64")

        # Floating point
        assert hasattr(DataType, "FP8")
        assert hasattr(DataType, "FP16")
        assert hasattr(DataType, "FP32")
        assert hasattr(DataType, "BF16")

        # Hybrid float
        assert hasattr(DataType, "HF4")
        assert hasattr(DataType, "HF8")

        # Unsigned integers
        assert hasattr(DataType, "UINT8")
        assert hasattr(DataType, "UINT16")
        assert hasattr(DataType, "UINT32")
        assert hasattr(DataType, "UINT64")

        # Boolean
        assert hasattr(DataType, "BOOL")

    def test_enum_values_are_unique(self):
        """Test that all enum values have unique integer values."""
        values = [
            DataType.INT4,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT4,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.FP4,
            DataType.FP8,
            DataType.FP16,
            DataType.FP32,
            DataType.BF16,
            DataType.HF4,
            DataType.HF8,
            DataType.BOOL,
        ]
        # Convert to int to compare underlying values
        int_values = [int(v) for v in values]
        assert len(int_values) == len(set(int_values)), "Enum values must be unique"

    def test_convenience_constants(self):
        """Test that convenience constants match DataType enum values."""
        assert DT_INT4 == DataType.INT4
        assert DT_INT8 == DataType.INT8
        assert DT_INT16 == DataType.INT16
        assert DT_INT32 == DataType.INT32
        assert DT_INT64 == DataType.INT64
        assert DT_UINT4 == DataType.UINT4
        assert DT_UINT8 == DataType.UINT8
        assert DT_UINT16 == DataType.UINT16
        assert DT_UINT32 == DataType.UINT32
        assert DT_UINT64 == DataType.UINT64
        assert DT_FP4 == DataType.FP4
        assert DT_FP8 == DataType.FP8
        assert DT_FP16 == DataType.FP16
        assert DT_FP32 == DataType.FP32
        assert DT_BF16 == DataType.BF16
        assert DT_HF4 == DataType.HF4
        assert DT_HF8 == DataType.HF8
        assert DT_BOOL == DataType.BOOL

    def test_convenience_constants_in_pypto_namespace(self):
        """Test that convenience constants are accessible from pypto module."""
        assert hasattr(pypto, "DT_INT4")
        assert hasattr(pypto, "DT_INT8")
        assert hasattr(pypto, "DT_INT16")
        assert hasattr(pypto, "DT_INT32")
        assert hasattr(pypto, "DT_INT64")
        assert hasattr(pypto, "DT_UINT4")
        assert hasattr(pypto, "DT_UINT8")
        assert hasattr(pypto, "DT_UINT16")
        assert hasattr(pypto, "DT_UINT32")
        assert hasattr(pypto, "DT_UINT64")
        assert hasattr(pypto, "DT_FP4")
        assert hasattr(pypto, "DT_FP8")
        assert hasattr(pypto, "DT_FP16")
        assert hasattr(pypto, "DT_FP32")
        assert hasattr(pypto, "DT_BF16")
        assert hasattr(pypto, "DT_HF4")
        assert hasattr(pypto, "DT_HF8")
        assert hasattr(pypto, "DT_BOOL")
        assert pypto.DT_INT32 == DataType.INT32


class TestDataTypeBit:
    """Test get_dtype_bit() function."""

    def test_4bit_types(self):
        """Test data types that are 4 bits."""
        assert get_dtype_bit(pypto.DT_INT4) == 4
        assert get_dtype_bit(pypto.DT_UINT4) == 4
        assert get_dtype_bit(pypto.DT_FP4) == 4
        assert get_dtype_bit(pypto.DT_HF4) == 4

    def test_8bit_types(self):
        """Test data types that are 8 bits."""
        assert get_dtype_bit(pypto.DT_INT8) == 8
        assert get_dtype_bit(pypto.DT_UINT8) == 8
        assert get_dtype_bit(pypto.DT_FP8) == 8
        assert get_dtype_bit(pypto.DT_HF8) == 8
        assert get_dtype_bit(pypto.DT_BOOL) == 8

    def test_16bit_types(self):
        """Test data types that are 16 bits."""
        assert get_dtype_bit(pypto.DT_INT16) == 16
        assert get_dtype_bit(pypto.DT_UINT16) == 16
        assert get_dtype_bit(pypto.DT_FP16) == 16
        assert get_dtype_bit(pypto.DT_BF16) == 16

    def test_32bit_types(self):
        """Test data types that are 32 bits."""
        assert get_dtype_bit(pypto.DT_INT32) == 32
        assert get_dtype_bit(pypto.DT_UINT32) == 32
        assert get_dtype_bit(pypto.DT_FP32) == 32

    def test_64bit_types(self):
        """Test data types that are 64 bits."""
        assert get_dtype_bit(pypto.DT_INT64) == 64
        assert get_dtype_bit(pypto.DT_UINT64) == 64


class TestDataTypeString:
    """Test dtype_to_string() function."""

    def test_signed_integer_strings(self):
        """Test string representation of signed integer types."""
        assert dtype_to_string(pypto.DT_INT4) == "int4"
        assert dtype_to_string(pypto.DT_INT8) == "int8"
        assert dtype_to_string(pypto.DT_INT16) == "int16"
        assert dtype_to_string(pypto.DT_INT32) == "int32"
        assert dtype_to_string(pypto.DT_INT64) == "int64"

    def test_unsigned_integer_strings(self):
        """Test string representation of unsigned integer types."""
        assert dtype_to_string(pypto.DT_UINT4) == "uint4"
        assert dtype_to_string(pypto.DT_UINT8) == "uint8"
        assert dtype_to_string(pypto.DT_UINT16) == "uint16"
        assert dtype_to_string(pypto.DT_UINT32) == "uint32"
        assert dtype_to_string(pypto.DT_UINT64) == "uint64"

    def test_floating_point_strings(self):
        """Test string representation of floating point types."""
        assert dtype_to_string(pypto.DT_FP4) == "fp4"
        assert dtype_to_string(pypto.DT_FP8) == "fp8"
        assert dtype_to_string(pypto.DT_FP16) == "fp16"
        assert dtype_to_string(pypto.DT_FP32) == "fp32"
        assert dtype_to_string(pypto.DT_BF16) == "bfloat16"

    def test_hybrid_float_strings(self):
        """Test string representation of hybrid float types."""
        assert dtype_to_string(pypto.DT_HF4) == "hf4"
        assert dtype_to_string(pypto.DT_HF8) == "hf8"

    def test_bool_string(self):
        """Test string representation of boolean type."""
        assert dtype_to_string(pypto.DT_BOOL) == "bool"


class TestDataTypePredicates:
    """Test type checking predicate functions."""

    def test_is_float(self):
        """Test is_float() correctly identifies floating point types."""
        # Floating point types
        assert is_float(pypto.DT_FP4) is True
        assert is_float(pypto.DT_FP8) is True
        assert is_float(pypto.DT_FP16) is True
        assert is_float(pypto.DT_FP32) is True
        assert is_float(pypto.DT_BF16) is True
        assert is_float(pypto.DT_HF4) is True
        assert is_float(pypto.DT_HF8) is True

        # Non-floating point types
        assert is_float(pypto.DT_INT8) is False
        assert is_float(pypto.DT_INT32) is False
        assert is_float(pypto.DT_UINT8) is False
        assert is_float(pypto.DT_BOOL) is False

    def test_is_signed_int(self):
        """Test is_signed_int() correctly identifies signed integer types."""
        # Signed integer types
        assert is_signed_int(pypto.DT_INT4) is True
        assert is_signed_int(pypto.DT_INT8) is True
        assert is_signed_int(pypto.DT_INT16) is True
        assert is_signed_int(pypto.DT_INT32) is True
        assert is_signed_int(pypto.DT_INT64) is True

        # Non-signed integer types
        assert is_signed_int(pypto.DT_UINT8) is False
        assert is_signed_int(pypto.DT_FP32) is False
        assert is_signed_int(pypto.DT_BOOL) is False

    def test_is_unsigned_int(self):
        """Test is_unsigned_int() correctly identifies unsigned integer types."""
        # Unsigned integer types
        assert is_unsigned_int(pypto.DT_UINT4) is True
        assert is_unsigned_int(pypto.DT_UINT8) is True
        assert is_unsigned_int(pypto.DT_UINT16) is True
        assert is_unsigned_int(pypto.DT_UINT32) is True
        assert is_unsigned_int(pypto.DT_UINT64) is True

        # Non-unsigned integer types
        assert is_unsigned_int(pypto.DT_INT8) is False
        assert is_unsigned_int(pypto.DT_FP32) is False
        assert is_unsigned_int(pypto.DT_BOOL) is False

    def test_is_int(self):
        """Test is_int() correctly identifies any integer types."""
        # Integer types (both signed and unsigned)
        assert is_int(pypto.DT_INT4) is True
        assert is_int(pypto.DT_INT8) is True
        assert is_int(pypto.DT_INT16) is True
        assert is_int(pypto.DT_INT32) is True
        assert is_int(pypto.DT_INT64) is True
        assert is_int(pypto.DT_UINT4) is True
        assert is_int(pypto.DT_UINT8) is True
        assert is_int(pypto.DT_UINT16) is True
        assert is_int(pypto.DT_UINT32) is True
        assert is_int(pypto.DT_UINT64) is True

        # Non-integer types
        assert is_int(pypto.DT_FP4) is False
        assert is_int(pypto.DT_FP8) is False
        assert is_int(pypto.DT_FP16) is False
        assert is_int(pypto.DT_FP32) is False
        assert is_int(pypto.DT_BF16) is False
        assert is_int(pypto.DT_HF4) is False
        assert is_int(pypto.DT_HF8) is False
        assert is_int(pypto.DT_BOOL) is False

    def test_type_predicates_mutual_exclusion(self):
        """Test that signed, unsigned, and floating point are mutually exclusive."""
        all_types = [
            DT_INT4,
            DT_INT8,
            DT_INT16,
            DT_INT32,
            DT_INT64,
            DT_FP4,
            DT_FP8,
            DT_FP16,
            DT_FP32,
            DT_BF16,
            DT_HF4,
            DT_HF8,
            DT_UINT4,
            DT_UINT8,
            DT_UINT16,
            DT_UINT32,
            DT_UINT64,
            DT_BOOL,
        ]

        for dtype in all_types:
            # A type should not be both signed integer and unsigned integer
            if is_signed_int(dtype):
                assert not is_unsigned_int(dtype)

            # A type should not be both integer and floating point
            if is_int(dtype):
                assert not is_float(dtype)


class TestDataTypeIntegration:
    """Integration tests for DataType system."""

    all_types: list[DataType] = [
        pypto.DT_INT4,
        pypto.DT_INT8,
        pypto.DT_INT16,
        pypto.DT_INT32,
        pypto.DT_INT64,
        pypto.DT_UINT4,
        pypto.DT_UINT8,
        pypto.DT_UINT16,
        pypto.DT_UINT32,
        pypto.DT_UINT64,
        pypto.DT_FP4,
        pypto.DT_FP8,
        pypto.DT_FP16,
        pypto.DT_FP32,
        pypto.DT_BF16,
        pypto.DT_HF4,
        pypto.DT_HF8,
        pypto.DT_BOOL,
    ]

    def test_all_types_have_bit_size(self):
        """Test that all data types have a valid bit size."""

        for dtype in self.all_types:
            bit_size = get_dtype_bit(dtype)
            assert bit_size > 0, f"Type {dtype_to_string(dtype)} should have positive bit size"
            assert bit_size in [4, 8, 16, 32, 64], f"Type {dtype_to_string(dtype)} should have valid bit size"

    def test_all_types_have_string_representation(self):
        """Test that all data types have a valid string representation."""

        for dtype in self.all_types:
            string_repr = dtype_to_string(dtype)
            assert string_repr != "unknown", f"Type {dtype} should have valid string representation"
            assert len(string_repr) > 0, f"Type {dtype} should have non-empty string representation"

    def test_all_types_classified(self):
        """Test that all data types are classified as either integer, float, or bool."""

        for dtype in self.all_types:
            is_integer = is_int(dtype)
            is_floating = is_float(dtype)
            is_boolean = dtype == pypto.DT_BOOL

            # Each type should be classified as at least one category
            # (bool is a special case that's neither int nor float in this classification)
            assert is_integer or is_floating or is_boolean, (
                f"Type {dtype_to_string(dtype)} should be classified as int, float, or bool"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
