
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void fa_perf_kernel(__gm__ half* q_0, __gm__ half* k_0, __gm__ half* v_0, __gm__ half* o_0, __gm__ float* qk_buf_0, __gm__ half* p_buf_0, __gm__ float* pv_buf_0, int32_t Sq, int32_t D, int32_t Skv, int32_t SqFifo, __gm__ int64_t* ffts_addr)
{
    set_ffts_base_addr((unsigned long)ffts_addr);

    using q_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using q_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using q_0GlobalType = GlobalTensor<half, q_0GlobalShapeDim5, q_0GlobalStrideDim5>;
    q_0GlobalType q_0Global(q_0);

    using k_0GlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using k_0GlobalStrideDim5 = Stride<128, 128, 128, 1, 128>;
    using k_0GlobalType = GlobalTensor<half, k_0GlobalShapeDim5, k_0GlobalStrideDim5, Layout::DN>;
    k_0GlobalType k_0Global(k_0);

    using v_0GlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using v_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using v_0GlobalType = GlobalTensor<half, v_0GlobalShapeDim5, v_0GlobalStrideDim5>;
    v_0GlobalType v_0Global(v_0);

    using o_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using o_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using o_0GlobalType = GlobalTensor<half, o_0GlobalShapeDim5, o_0GlobalStrideDim5>;
    o_0GlobalType o_0Global(o_0);

    using qk_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using qk_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using qk_buf_0GlobalType = GlobalTensor<float, qk_buf_0GlobalShapeDim5, qk_buf_0GlobalStrideDim5>;
    qk_buf_0GlobalType qk_buf_0Global(qk_buf_0);

    using p_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using p_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using p_buf_0GlobalType = GlobalTensor<half, p_buf_0GlobalShapeDim5, p_buf_0GlobalStrideDim5>;
    p_buf_0GlobalType p_buf_0Global(p_buf_0);

    using pv_buf_0GlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using pv_buf_0GlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using pv_buf_0GlobalType = GlobalTensor<float, pv_buf_0GlobalShapeDim5, pv_buf_0GlobalStrideDim5>;
    pv_buf_0GlobalType pv_buf_0Global(pv_buf_0);

    #if defined(__DAV_CUBE__)
    // Tile declarations (Cube)
    using q_mat_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_0_0Type q_mat_0_0(64, 128);
    TASSIGN(q_mat_0_0, 0x0);
    using q_mat_1_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_1_0Type q_mat_1_0(64, 128);
    TASSIGN(q_mat_1_0, 0x4000);
    using k_mat_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_0_0Type k_mat_0_0(128, 128);
    TASSIGN(k_mat_0_0, 0x8000);
    using k_mat_1_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_1_0Type k_mat_1_0(128, 128);
    TASSIGN(k_mat_1_0, 0x10000);
    using p_mat_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_0_0Type p_mat_0_0(64, 128);
    TASSIGN(p_mat_0_0, 0x18000);
    using p_mat_1_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_1_0Type p_mat_1_0(64, 128);
    TASSIGN(p_mat_1_0, 0x1c000);
    using v_mat_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_0_0Type v_mat_0_0(128, 128);
    TASSIGN(v_mat_0_0, 0x20000);
    using v_mat_1_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_1_0Type v_mat_1_0(128, 128);
    TASSIGN(v_mat_1_0, 0x28000);
    using left_0_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_0_0Type left_0_0(64, 128);
    TASSIGN(left_0_0, 0x0);
    using left_1_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_1_0Type left_1_0(64, 128);
    TASSIGN(left_1_0, 0x4000);
    using right_0_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_0_0Type right_0_0(128, 128);
    TASSIGN(right_0_0, 0x0);
    using right_1_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_1_0Type right_1_0(128, 128);
    TASSIGN(right_1_0, 0x8000);
    using acc_0_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_0_0Type acc_0_0(64, 128);
    TASSIGN(acc_0_0, 0x0);
    using acc_1_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_1_0Type acc_1_0(64, 128);
    TASSIGN(acc_1_0, 0x8000);
    using _tuple_tmp_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_0Type _tuple_tmp_0_0(64, 128);
    TASSIGN(_tuple_tmp_0_0, 0x0);
    using _tuple_tmp_0_1Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_1Type _tuple_tmp_0_1(64, 128);
    TASSIGN(_tuple_tmp_0_1, 0x4000);
    using _tuple_tmp_0_2Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    _tuple_tmp_0_2Type _tuple_tmp_0_2(128, 128);
    TASSIGN(_tuple_tmp_0_2, 0x8000);
    using _tuple_tmp_0_3Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    _tuple_tmp_0_3Type _tuple_tmp_0_3(128, 128);
    TASSIGN(_tuple_tmp_0_3, 0x10000);
    using _tuple_tmp_0_4Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_4Type _tuple_tmp_0_4(64, 128);
    TASSIGN(_tuple_tmp_0_4, 0x18000);
    using _tuple_tmp_0_5Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_5Type _tuple_tmp_0_5(64, 128);
    TASSIGN(_tuple_tmp_0_5, 0x1c000);
    using _tuple_tmp_0_6Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_6Type _tuple_tmp_0_6(128, 128);
    TASSIGN(_tuple_tmp_0_6, 0x20000);
    using _tuple_tmp_0_7Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_7Type _tuple_tmp_0_7(128, 128);
    TASSIGN(_tuple_tmp_0_7, 0x28000);
    using _tuple_tmp_0_8Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_8Type _tuple_tmp_0_8(64, 128);
    TASSIGN(_tuple_tmp_0_8, 0x0);
    using _tuple_tmp_0_9Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    _tuple_tmp_0_9Type _tuple_tmp_0_9(64, 128);
    TASSIGN(_tuple_tmp_0_9, 0x4000);
    using _tuple_tmp_0_10Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    _tuple_tmp_0_10Type _tuple_tmp_0_10(128, 128);
    TASSIGN(_tuple_tmp_0_10, 0x0);
    using _tuple_tmp_0_11Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    _tuple_tmp_0_11Type _tuple_tmp_0_11(128, 128);
    TASSIGN(_tuple_tmp_0_11, 0x8000);
    using _tuple_tmp_0_12Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    _tuple_tmp_0_12Type _tuple_tmp_0_12(64, 128);
    TASSIGN(_tuple_tmp_0_12, 0x0);
    using _tuple_tmp_0_13Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    _tuple_tmp_0_13Type _tuple_tmp_0_13(64, 128);
    TASSIGN(_tuple_tmp_0_13, 0x8000);
    using q_mat_buf_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_buf_0_0Type q_mat_buf_0_0(64, 128);
    TASSIGN(q_mat_buf_0_0, 0x0);
    using q_mat_buf_0_1Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_buf_0_1Type q_mat_buf_0_1(64, 128);
    TASSIGN(q_mat_buf_0_1, 0x4000);
    using k_mat_buf_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_buf_0_0Type k_mat_buf_0_0(128, 128);
    TASSIGN(k_mat_buf_0_0, 0x8000);
    using k_mat_buf_0_1Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_buf_0_1Type k_mat_buf_0_1(128, 128);
    TASSIGN(k_mat_buf_0_1, 0x10000);
    using p_mat_buf_0_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_buf_0_0Type p_mat_buf_0_0(64, 128);
    TASSIGN(p_mat_buf_0_0, 0x18000);
    using p_mat_buf_0_1Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_buf_0_1Type p_mat_buf_0_1(64, 128);
    TASSIGN(p_mat_buf_0_1, 0x1c000);
    using v_mat_buf_0_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_buf_0_0Type v_mat_buf_0_0(128, 128);
    TASSIGN(v_mat_buf_0_0, 0x20000);
    using v_mat_buf_0_1Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_buf_0_1Type v_mat_buf_0_1(128, 128);
    TASSIGN(v_mat_buf_0_1, 0x28000);
    using left_buf_0_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_buf_0_0Type left_buf_0_0(64, 128);
    TASSIGN(left_buf_0_0, 0x0);
    using left_buf_0_1Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_buf_0_1Type left_buf_0_1(64, 128);
    TASSIGN(left_buf_0_1, 0x4000);
    using right_buf_0_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_buf_0_0Type right_buf_0_0(128, 128);
    TASSIGN(right_buf_0_0, 0x0);
    using right_buf_0_1Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_buf_0_1Type right_buf_0_1(128, 128);
    TASSIGN(right_buf_0_1, 0x8000);
    using acc_buf_0_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_buf_0_0Type acc_buf_0_0(64, 128);
    TASSIGN(acc_buf_0_0, 0x0);
    using acc_buf_0_1Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_buf_0_1Type acc_buf_0_1(64, 128);
    TASSIGN(acc_buf_0_1, 0x8000);
    #endif  // __DAV_CUBE__

    #if defined(__DAV_VEC__)
    // Tile declarations (Vector)
    using qk_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    qk_vec_0Type qk_vec_0(64, 128);
    TASSIGN(qk_vec_0, 0x0);
    using tmp_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    tmp_vec_0Type tmp_vec_0(64, 128);
    TASSIGN(tmp_vec_0, 0x8000);
    using p_f16_0Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    p_f16_0Type p_f16_0(64, 128);
    TASSIGN(p_f16_0, 0x10000);
    using reduce_dst_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dst_0Type reduce_dst_0(64, 1);
    TASSIGN(reduce_dst_0, 0x14000);
    using reduce_dst_rm_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dst_rm_0Type reduce_dst_rm_0(1, 64);
    TASSIGN(reduce_dst_rm_0, 0x14000);
    using gmax_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    gmax_0_0Type gmax_0_0(64, 1);
    TASSIGN(gmax_0_0, 0x14100);
    using gmax_1_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    gmax_1_0Type gmax_1_0(64, 1);
    TASSIGN(gmax_1_0, 0x14200);
    using gmax_rm_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gmax_rm_0_0Type gmax_rm_0_0(1, 64);
    TASSIGN(gmax_rm_0_0, 0x14100);
    using gmax_rm_1_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gmax_rm_1_0Type gmax_rm_1_0(1, 64);
    TASSIGN(gmax_rm_1_0, 0x14200);
    using global_max_buf_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_buf_0_0Type global_max_buf_0_0(64, 1);
    TASSIGN(global_max_buf_0_0, 0x14100);
    using global_max_buf_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_buf_0_1Type global_max_buf_0_1(64, 1);
    TASSIGN(global_max_buf_0_1, 0x14200);
    using global_max_rm_buf_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_rm_buf_0_0Type global_max_rm_buf_0_0(1, 64);
    TASSIGN(global_max_rm_buf_0_0, 0x14100);
    using global_max_rm_buf_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_rm_buf_0_1Type global_max_rm_buf_0_1(1, 64);
    TASSIGN(global_max_rm_buf_0_1, 0x14200);
    using gsum_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_0_0Type gsum_0_0(64, 1);
    TASSIGN(gsum_0_0, 0x14300);
    using gsum_1_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_1_0Type gsum_1_0(64, 1);
    TASSIGN(gsum_1_0, 0x14400);
    using gsum_rm_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_rm_0_0Type gsum_rm_0_0(1, 64);
    TASSIGN(gsum_rm_0_0, 0x14300);
    using gsum_rm_1_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_rm_1_0Type gsum_rm_1_0(1, 64);
    TASSIGN(gsum_rm_1_0, 0x14400);
    using global_sum_buf_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_buf_0_0Type global_sum_buf_0_0(64, 1);
    TASSIGN(global_sum_buf_0_0, 0x14300);
    using global_sum_buf_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_buf_0_1Type global_sum_buf_0_1(64, 1);
    TASSIGN(global_sum_buf_0_1, 0x14400);
    using global_sum_rm_buf_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_rm_buf_0_0Type global_sum_rm_buf_0_0(1, 64);
    TASSIGN(global_sum_rm_buf_0_0, 0x14300);
    using global_sum_rm_buf_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_rm_buf_0_1Type global_sum_rm_buf_0_1(1, 64);
    TASSIGN(global_sum_rm_buf_0_1, 0x14400);
    using ec0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    ec0_0Type ec0_0(64, 1);
    TASSIGN(ec0_0, 0x14500);
    using ec0_rm_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    ec0_rm_0Type ec0_rm_0(1, 64);
    TASSIGN(ec0_rm_0, 0x14500);
    using ec1_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    ec1_0Type ec1_0(64, 1);
    TASSIGN(ec1_0, 0x14600);
    using ec1_rm_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    ec1_rm_0Type ec1_rm_0(1, 64);
    TASSIGN(ec1_rm_0, 0x14600);
    using _tuple_tmp_1_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    _tuple_tmp_1_0Type _tuple_tmp_1_0(64, 1);
    TASSIGN(_tuple_tmp_1_0, 0x14500);
    using _tuple_tmp_1_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    _tuple_tmp_1_1Type _tuple_tmp_1_1(64, 1);
    TASSIGN(_tuple_tmp_1_1, 0x14600);
    using _tuple_tmp_1_2Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    _tuple_tmp_1_2Type _tuple_tmp_1_2(1, 64);
    TASSIGN(_tuple_tmp_1_2, 0x14500);
    using _tuple_tmp_1_3Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    _tuple_tmp_1_3Type _tuple_tmp_1_3(1, 64);
    TASSIGN(_tuple_tmp_1_3, 0x14600);
    using exp_corr_fifo_0_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_fifo_0_0Type exp_corr_fifo_0_0(64, 1);
    TASSIGN(exp_corr_fifo_0_0, 0x14500);
    using exp_corr_fifo_0_1Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_fifo_0_1Type exp_corr_fifo_0_1(64, 1);
    TASSIGN(exp_corr_fifo_0_1, 0x14600);
    using exp_corr_rm_fifo_0_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_rm_fifo_0_0Type exp_corr_rm_fifo_0_0(1, 64);
    TASSIGN(exp_corr_rm_fifo_0_0, 0x14500);
    using exp_corr_rm_fifo_0_1Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    exp_corr_rm_fifo_0_1Type exp_corr_rm_fifo_0_1(1, 64);
    TASSIGN(exp_corr_rm_fifo_0_1, 0x14600);
    using running_o_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    running_o_0Type running_o_0(64, 128);
    TASSIGN(running_o_0, 0x14700);
    using pv_vec_0Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    pv_vec_0Type pv_vec_0(64, 128);
    TASSIGN(pv_vec_0, 0x1c700);
    using o_f16_0Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    o_f16_0Type o_f16_0(64, 128);
    TASSIGN(o_f16_0, 0x24700);
    using global_max_cur_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_cur_0Type global_max_cur_0(64, 1);
    TASSIGN(global_max_cur_0, 0x14100);
    using global_max_rm_cur_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_max_rm_cur_0Type global_max_rm_cur_0(1, 64);
    TASSIGN(global_max_rm_cur_0, 0x14100);
    using global_sum_cur_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_cur_0Type global_sum_cur_0(64, 1);
    TASSIGN(global_sum_cur_0, 0x14300);
    using global_sum_rm_cur_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    global_sum_rm_cur_0Type global_sum_rm_cur_0(1, 64);
    TASSIGN(global_sum_rm_cur_0, 0x14300);
    #endif  // __DAV_VEC__


    // Function body
    auto sq_dim_0 = Sq;
    auto skv_dim_0 = Skv;
    auto sq_tiles_0 = ((sq_dim_0 + 127) / 128);
    auto skv_tiles_0 = ((skv_dim_0 + 127) / 128);
    auto num_cores_0 = (int32_t)(get_block_num());
    auto core_id_0 = (int32_t)(get_block_idx());
    #if defined(__DAV_CUBE__)
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    auto _ctx_sq_off_0 = 0;
    auto _ctx_row_off_0 = 0;
    auto _ctx_task_id_0 = 0;
    auto _ctx_q_count_0 = 0;
    auto _ctx_buf_idx_0 = 0;
    auto _ctx_l0ab_idx_0 = 0;
    auto _ctx_l0c_idx_0 = 0;
    auto _ctx_core_id_0 = core_id_0;
    // Loop-carried values initialization
    auto _ctx_buf_idx_iter_1 = _ctx_buf_idx_0;
    auto _ctx_l0ab_idx_iter_1 = _ctx_l0ab_idx_0;
    auto _ctx_l0c_idx_iter_1 = _ctx_l0c_idx_0;
    auto _ctx_q_count_iter_1 = _ctx_q_count_0;
    auto _ctx_row_off_iter_1 = _ctx_row_off_0;
    auto _ctx_sq_off_iter_1 = _ctx_sq_off_0;
    auto _ctx_task_id_iter_1 = _ctx_task_id_0;

    for (uint64_t qi_0 = core_id_0; qi_0 < sq_tiles_0; qi_0 += num_cores_0) {
        auto _ctx_sq_off_3 = (qi_0 * 128);
        // Loop-carried values initialization
        auto _ctx_buf_idx_iter_3 = _ctx_buf_idx_iter_1;
        auto _ctx_l0ab_idx_iter_3 = _ctx_l0ab_idx_iter_1;
        auto _ctx_l0c_idx_iter_3 = _ctx_l0c_idx_iter_1;
        auto _ctx_q_count_iter_3 = _ctx_q_count_iter_1;
        auto _ctx_row_off_iter_3 = _ctx_row_off_iter_1;
        auto _ctx_task_id_iter_3 = _ctx_task_id_iter_1;

        for (uint64_t row_idx_0 = 0; row_idx_0 < 2; row_idx_0 += 1) {
            auto _ctx_row_off_5 = (row_idx_0 * 64);
            // Loop-carried values initialization
            auto _ctx_buf_idx_iter_5 = _ctx_buf_idx_iter_3;
            auto _ctx_l0ab_idx_iter_5 = _ctx_l0ab_idx_iter_3;
            auto _ctx_l0c_idx_iter_5 = _ctx_l0c_idx_iter_3;
            auto _ctx_task_id_iter_5 = _ctx_task_id_iter_3;

            for (uint64_t pre_0 = 0; pre_0 < 1; pre_0 += 1) {
                auto _ctx_task_id_7 = pre_0;
                auto _ctx_buf_idx_7 = (((_ctx_q_count_iter_3 * skv_tiles_0) + pre_0) % 2);
                auto q_mat_idx_0 = (_ctx_q_count_iter_3 % 2);
                auto qk_fifo_slot_0 = (_ctx_task_id_7 % 2);
                auto skv_off_0 = (_ctx_task_id_7 * 128);
                int64_t _tidx_0_0;

                if ((_ctx_buf_idx_7 == 0)) {
                    _tidx_0_0 = 0;
                } else {
                    _tidx_0_0 = 1;
                }

                wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_0_0);
                if ((_ctx_task_id_7 == 0)) {
                    using _tidx_1_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                    _tidx_1_0Type _tidx_1_0(64, 128);

                    if ((q_mat_idx_0 == 0)) {
                        TASSIGN(_tidx_1_0, 0x0);
                    } else {
                        TASSIGN(_tidx_1_0, 0x4000);
                    }

                    TASSIGN(q_0Global, q_0 + ((_ctx_sq_off_3 + _ctx_row_off_5) * D + 0));
                    TLOAD(_tidx_1_0, q_0Global);
                }

                using _tidx_2_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_2_0Type _tidx_2_0(128, 128);

                if ((_ctx_buf_idx_7 == 0)) {
                    TASSIGN(_tidx_2_0, 0x8000);
                } else {
                    TASSIGN(_tidx_2_0, 0x10000);
                }

                TASSIGN(k_0Global, k_0 + (skv_off_0 * D + 0));
                TLOAD(_tidx_2_0, k_0Global);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                int64_t _tidx_3_0;

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    _tidx_3_0 = 0;
                } else {
                    _tidx_3_0 = 1;
                }

                wait_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_3_0);
                using _tidx_4_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_4_0Type _tidx_4_0(64, 128);

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_4_0, 0x0);
                } else {
                    TASSIGN(_tidx_4_0, 0x4000);
                }

                using _tidx_5_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_5_0Type _tidx_5_0(64, 128);

                if ((q_mat_idx_0 == 0)) {
                    TASSIGN(_tidx_5_0, 0x0);
                } else {
                    TASSIGN(_tidx_5_0, 0x4000);
                }

                TMOV(_tidx_4_0, _tidx_5_0);
                using _tidx_6_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_6_0Type _tidx_6_0(128, 128);

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_6_0, 0x0);
                } else {
                    TASSIGN(_tidx_6_0, 0x8000);
                }

                using _tidx_7_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_7_0Type _tidx_7_0(128, 128);

                if ((_ctx_buf_idx_7 == 0)) {
                    TASSIGN(_tidx_7_0, 0x8000);
                } else {
                    TASSIGN(_tidx_7_0, 0x10000);
                }

                TMOV(_tidx_6_0, _tidx_7_0);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                int64_t _tidx_8_0;

                if ((_ctx_buf_idx_7 == 0)) {
                    _tidx_8_0 = 0;
                } else {
                    _tidx_8_0 = 1;
                }

                set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_8_0);
                int64_t _tidx_9_0;

                if ((_ctx_l0c_idx_iter_5 == 0)) {
                    _tidx_9_0 = 0;
                } else {
                    _tidx_9_0 = 1;
                }

                wait_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_9_0);
                using _tidx_10_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                _tidx_10_0Type _tidx_10_0(64, 128);

                if ((_ctx_l0c_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_10_0, 0x0);
                } else {
                    TASSIGN(_tidx_10_0, 0x8000);
                }

                using _tidx_11_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_11_0Type _tidx_11_0(64, 128);

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_11_0, 0x0);
                } else {
                    TASSIGN(_tidx_11_0, 0x4000);
                }

                using _tidx_12_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_12_0Type _tidx_12_0(128, 128);

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_12_0, 0x0);
                } else {
                    TASSIGN(_tidx_12_0, 0x8000);
                }

                TMATMUL(_tidx_10_0, _tidx_11_0, _tidx_12_0);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                int64_t _tidx_13_0;

                if ((_ctx_l0ab_idx_iter_5 == 0)) {
                    _tidx_13_0 = 0;
                } else {
                    _tidx_13_0 = 1;
                }

                set_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_13_0);
                using _tidx_14_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                _tidx_14_0Type _tidx_14_0(64, 128);

                if ((_ctx_l0c_idx_iter_5 == 0)) {
                    TASSIGN(_tidx_14_0, 0x0);
                } else {
                    TASSIGN(_tidx_14_0, 0x8000);
                }

                TASSIGN(qk_buf_0Global, qk_buf_0 + ((((qk_fifo_slot_0 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * Skv + skv_off_0));
                TSTORE(qk_buf_0Global, _tidx_14_0);
                int64_t _tidx_15_0;

                if ((_ctx_l0c_idx_iter_5 == 0)) {
                    _tidx_15_0 = 0;
                } else {
                    _tidx_15_0 = 1;
                }

                set_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_15_0);
                auto _ctx_l0ab_idx_7 = (1 - _ctx_l0ab_idx_iter_5);
                auto _ctx_l0c_idx_7 = (1 - _ctx_l0c_idx_iter_5);
                int64_t _tidx_16_0;

                if ((qk_fifo_slot_0 == 0)) {
                    _tidx_16_0 = 0;
                } else {
                    _tidx_16_0 = 1;
                }

                if (_tidx_16_0 == 0) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 0));
                if (_tidx_16_0 == 1) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 1));
                if (_tidx_16_0 == 2) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 2));
                _ctx_buf_idx_iter_5 = _ctx_buf_idx_7;
                _ctx_l0ab_idx_iter_5 = _ctx_l0ab_idx_7;
                _ctx_l0c_idx_iter_5 = _ctx_l0c_idx_7;
                _ctx_task_id_iter_5 = _ctx_task_id_7;
            }

            // Loop-carried values initialization
            auto _ctx_buf_idx_iter_8 = _ctx_buf_idx_iter_5;
            auto _ctx_l0ab_idx_iter_8 = _ctx_l0ab_idx_iter_5;
            auto _ctx_l0c_idx_iter_8 = _ctx_l0c_idx_iter_5;
            auto _ctx_task_id_iter_8 = _ctx_task_id_iter_5;

            for (uint64_t ki_0 = 0; ki_0 < skv_tiles_0; ki_0 += 1) {
                auto next_ki_0 = (ki_0 + 1);
                int64_t _ctx_buf_idx_11;
                int64_t _ctx_l0ab_idx_11;
                int64_t _ctx_l0c_idx_11;
                int64_t _ctx_task_id_11;

                if ((next_ki_0 < skv_tiles_0)) {
                    auto _ctx_task_id_10 = next_ki_0;
                    auto _ctx_buf_idx_10 = (((_ctx_q_count_iter_3 * skv_tiles_0) + next_ki_0) % 2);
                    auto q_mat_idx_1 = (_ctx_q_count_iter_3 % 2);
                    auto qk_fifo_slot_1 = (_ctx_task_id_10 % 2);
                    auto skv_off_1 = (_ctx_task_id_10 * 128);
                    int64_t _tidx_17_0;

                    if ((_ctx_buf_idx_10 == 0)) {
                        _tidx_17_0 = 0;
                    } else {
                        _tidx_17_0 = 1;
                    }

                    wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_17_0);
                    if ((_ctx_task_id_10 == 0)) {
                        using _tidx_18_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                        _tidx_18_0Type _tidx_18_0(64, 128);

                        if ((q_mat_idx_1 == 0)) {
                            TASSIGN(_tidx_18_0, 0x0);
                        } else {
                            TASSIGN(_tidx_18_0, 0x4000);
                        }

                        TASSIGN(q_0Global, q_0 + ((_ctx_sq_off_3 + _ctx_row_off_5) * D + 0));
                        TLOAD(_tidx_18_0, q_0Global);
                    }

                    using _tidx_19_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                    _tidx_19_0Type _tidx_19_0(128, 128);

                    if ((_ctx_buf_idx_10 == 0)) {
                        TASSIGN(_tidx_19_0, 0x8000);
                    } else {
                        TASSIGN(_tidx_19_0, 0x10000);
                    }

                    TASSIGN(k_0Global, k_0 + (skv_off_1 * D + 0));
                    TLOAD(_tidx_19_0, k_0Global);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    int64_t _tidx_20_0;

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        _tidx_20_0 = 0;
                    } else {
                        _tidx_20_0 = 1;
                    }

                    wait_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_20_0);
                    using _tidx_21_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                    _tidx_21_0Type _tidx_21_0(64, 128);

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_21_0, 0x0);
                    } else {
                        TASSIGN(_tidx_21_0, 0x4000);
                    }

                    using _tidx_22_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                    _tidx_22_0Type _tidx_22_0(64, 128);

                    if ((q_mat_idx_1 == 0)) {
                        TASSIGN(_tidx_22_0, 0x0);
                    } else {
                        TASSIGN(_tidx_22_0, 0x4000);
                    }

                    TMOV(_tidx_21_0, _tidx_22_0);
                    using _tidx_23_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                    _tidx_23_0Type _tidx_23_0(128, 128);

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_23_0, 0x0);
                    } else {
                        TASSIGN(_tidx_23_0, 0x8000);
                    }

                    using _tidx_24_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                    _tidx_24_0Type _tidx_24_0(128, 128);

                    if ((_ctx_buf_idx_10 == 0)) {
                        TASSIGN(_tidx_24_0, 0x8000);
                    } else {
                        TASSIGN(_tidx_24_0, 0x10000);
                    }

                    TMOV(_tidx_23_0, _tidx_24_0);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    int64_t _tidx_25_0;

                    if ((_ctx_buf_idx_10 == 0)) {
                        _tidx_25_0 = 0;
                    } else {
                        _tidx_25_0 = 1;
                    }

                    set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_25_0);
                    int64_t _tidx_26_0;

                    if ((_ctx_l0c_idx_iter_8 == 0)) {
                        _tidx_26_0 = 0;
                    } else {
                        _tidx_26_0 = 1;
                    }

                    wait_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_26_0);
                    using _tidx_27_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                    _tidx_27_0Type _tidx_27_0(64, 128);

                    if ((_ctx_l0c_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_27_0, 0x0);
                    } else {
                        TASSIGN(_tidx_27_0, 0x8000);
                    }

                    using _tidx_28_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                    _tidx_28_0Type _tidx_28_0(64, 128);

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_28_0, 0x0);
                    } else {
                        TASSIGN(_tidx_28_0, 0x4000);
                    }

                    using _tidx_29_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                    _tidx_29_0Type _tidx_29_0(128, 128);

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_29_0, 0x0);
                    } else {
                        TASSIGN(_tidx_29_0, 0x8000);
                    }

                    TMATMUL(_tidx_27_0, _tidx_28_0, _tidx_29_0);
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    int64_t _tidx_30_0;

                    if ((_ctx_l0ab_idx_iter_8 == 0)) {
                        _tidx_30_0 = 0;
                    } else {
                        _tidx_30_0 = 1;
                    }

                    set_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_30_0);
                    using _tidx_31_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                    _tidx_31_0Type _tidx_31_0(64, 128);

                    if ((_ctx_l0c_idx_iter_8 == 0)) {
                        TASSIGN(_tidx_31_0, 0x0);
                    } else {
                        TASSIGN(_tidx_31_0, 0x8000);
                    }

                    TASSIGN(qk_buf_0Global, qk_buf_0 + ((((qk_fifo_slot_1 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * Skv + skv_off_1));
                    TSTORE(qk_buf_0Global, _tidx_31_0);
                    int64_t _tidx_32_0;

                    if ((_ctx_l0c_idx_iter_8 == 0)) {
                        _tidx_32_0 = 0;
                    } else {
                        _tidx_32_0 = 1;
                    }

                    set_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_32_0);
                    auto _ctx_l0ab_idx_10 = (1 - _ctx_l0ab_idx_iter_8);
                    auto _ctx_l0c_idx_10 = (1 - _ctx_l0c_idx_iter_8);
                    int64_t _tidx_33_0;

                    if ((qk_fifo_slot_1 == 0)) {
                        _tidx_33_0 = 0;
                    } else {
                        _tidx_33_0 = 1;
                    }

                    if (_tidx_33_0 == 0) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 0));
                    if (_tidx_33_0 == 1) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 1));
                    if (_tidx_33_0 == 2) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 2));
                    _ctx_buf_idx_11 = _ctx_buf_idx_10;
                    _ctx_l0ab_idx_11 = _ctx_l0ab_idx_10;
                    _ctx_l0c_idx_11 = _ctx_l0c_idx_10;
                    _ctx_task_id_11 = _ctx_task_id_10;
                } else {
                    _ctx_buf_idx_11 = _ctx_buf_idx_iter_8;
                    _ctx_l0ab_idx_11 = _ctx_l0ab_idx_iter_8;
                    _ctx_l0c_idx_11 = _ctx_l0c_idx_iter_8;
                    _ctx_task_id_11 = _ctx_task_id_iter_8;
                }

                auto _ctx_task_id_12 = ki_0;
                auto _ctx_buf_idx_12 = (((_ctx_q_count_iter_3 * skv_tiles_0) + ki_0) % 2);
                auto q_mat_idx_2 = (_ctx_q_count_iter_3 % 2);
                auto pv_task_slot_0 = (_ctx_task_id_12 % 2);
                auto sv_off_0 = (_ctx_task_id_12 * 128);
                auto pv_fifo_slot_0 = (_ctx_task_id_12 % 2);
                int64_t _tidx_34_0;

                if ((_ctx_buf_idx_12 == 0)) {
                    _tidx_34_0 = 2;
                } else {
                    _tidx_34_0 = 3;
                }

                wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_34_0);
                using _tidx_35_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_35_0Type _tidx_35_0(128, 128);

                if ((_ctx_buf_idx_12 == 0)) {
                    TASSIGN(_tidx_35_0, 0x20000);
                } else {
                    TASSIGN(_tidx_35_0, 0x28000);
                }

                TASSIGN(v_0Global, v_0 + (sv_off_0 * D + 0));
                TLOAD(_tidx_35_0, v_0Global);
                int64_t _tidx_36_0;

                if ((pv_fifo_slot_0 == 0)) {
                    _tidx_36_0 = 2;
                } else {
                    _tidx_36_0 = 3;
                }

                if (_tidx_36_0 == 0) wait_flag_dev(0);
                if (_tidx_36_0 == 1) wait_flag_dev(1);
                if (_tidx_36_0 == 2) wait_flag_dev(2);
                if (_tidx_36_0 == 3) wait_flag_dev(3);
                if (_tidx_36_0 == 4) wait_flag_dev(4);
                using _tidx_37_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_37_0Type _tidx_37_0(64, 128);

                if ((_ctx_buf_idx_12 == 0)) {
                    TASSIGN(_tidx_37_0, 0x18000);
                } else {
                    TASSIGN(_tidx_37_0, 0x1c000);
                }

                TASSIGN(p_buf_0Global, p_buf_0 + ((((pv_fifo_slot_0 * sq_dim_0) + _ctx_sq_off_3) + _ctx_row_off_5) * Skv + sv_off_0));
                TLOAD(_tidx_37_0, p_buf_0Global);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                int64_t _tidx_38_0;

                if ((_ctx_l0ab_idx_11 == 0)) {
                    _tidx_38_0 = 0;
                } else {
                    _tidx_38_0 = 1;
                }

                wait_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_38_0);
                using _tidx_39_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_39_0Type _tidx_39_0(64, 128);

                if ((_ctx_l0ab_idx_11 == 0)) {
                    TASSIGN(_tidx_39_0, 0x0);
                } else {
                    TASSIGN(_tidx_39_0, 0x4000);
                }

                using _tidx_40_0Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_40_0Type _tidx_40_0(64, 128);

                if ((_ctx_buf_idx_12 == 0)) {
                    TASSIGN(_tidx_40_0, 0x18000);
                } else {
                    TASSIGN(_tidx_40_0, 0x1c000);
                }

                TMOV(_tidx_39_0, _tidx_40_0);
                using _tidx_41_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_41_0Type _tidx_41_0(128, 128);

                if ((_ctx_l0ab_idx_11 == 0)) {
                    TASSIGN(_tidx_41_0, 0x0);
                } else {
                    TASSIGN(_tidx_41_0, 0x8000);
                }

                using _tidx_42_0Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_42_0Type _tidx_42_0(128, 128);

                if ((_ctx_buf_idx_12 == 0)) {
                    TASSIGN(_tidx_42_0, 0x20000);
                } else {
                    TASSIGN(_tidx_42_0, 0x28000);
                }

                TMOV(_tidx_41_0, _tidx_42_0);
                int64_t _tidx_43_0;

                if ((_ctx_buf_idx_12 == 0)) {
                    _tidx_43_0 = 2;
                } else {
                    _tidx_43_0 = 3;
                }

                set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)_tidx_43_0);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                int64_t _tidx_44_0;

                if ((_ctx_l0c_idx_11 == 0)) {
                    _tidx_44_0 = 0;
                } else {
                    _tidx_44_0 = 1;
                }

                wait_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_44_0);
                using _tidx_45_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                _tidx_45_0Type _tidx_45_0(64, 128);

                if ((_ctx_l0c_idx_11 == 0)) {
                    TASSIGN(_tidx_45_0, 0x0);
                } else {
                    TASSIGN(_tidx_45_0, 0x8000);
                }

                using _tidx_46_0Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
                _tidx_46_0Type _tidx_46_0(64, 128);

                if ((_ctx_l0ab_idx_11 == 0)) {
                    TASSIGN(_tidx_46_0, 0x0);
                } else {
                    TASSIGN(_tidx_46_0, 0x4000);
                }

                using _tidx_47_0Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
                _tidx_47_0Type _tidx_47_0(128, 128);

                if ((_ctx_l0ab_idx_11 == 0)) {
                    TASSIGN(_tidx_47_0, 0x0);
                } else {
                    TASSIGN(_tidx_47_0, 0x8000);
                }

                TMATMUL(_tidx_45_0, _tidx_46_0, _tidx_47_0);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                int64_t _tidx_48_0;

                if ((_ctx_l0ab_idx_11 == 0)) {
                    _tidx_48_0 = 0;
                } else {
                    _tidx_48_0 = 1;
                }

                set_flag(PIPE_M, PIPE_MTE1, (event_t)_tidx_48_0);
                using _tidx_49_0Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
                _tidx_49_0Type _tidx_49_0(64, 128);

                if ((_ctx_l0c_idx_11 == 0)) {
                    TASSIGN(_tidx_49_0, 0x0);
                } else {
                    TASSIGN(_tidx_49_0, 0x8000);
                }

                TASSIGN(pv_buf_0Global, pv_buf_0 + (((((_ctx_core_id_0 * 512) + ((q_mat_idx_2 * 2) * 128)) + (pv_task_slot_0 * 128)) + _ctx_row_off_5) * D + 0));
                TSTORE(pv_buf_0Global, _tidx_49_0);
                int64_t _tidx_50_0;

                if ((_ctx_l0c_idx_11 == 0)) {
                    _tidx_50_0 = 0;
                } else {
                    _tidx_50_0 = 1;
                }

                set_flag(PIPE_FIX, PIPE_M, (event_t)_tidx_50_0);
                auto _ctx_l0ab_idx_12 = (1 - _ctx_l0ab_idx_11);
                auto _ctx_l0c_idx_12 = (1 - _ctx_l0c_idx_11);
                int64_t _tidx_51_0;

                if ((pv_task_slot_0 == 0)) {
                    _tidx_51_0 = 4;
                } else {
                    _tidx_51_0 = 5;
                }

                if (_tidx_51_0 == 0) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 0));
                if (_tidx_51_0 == 1) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 1));
                if (_tidx_51_0 == 2) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 2));
                if (_tidx_51_0 == 3) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 3));
                if (_tidx_51_0 == 4) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 4));
                if (_tidx_51_0 == 5) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 5));
                if (_tidx_51_0 == 6) ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, 6));
                _ctx_buf_idx_iter_8 = _ctx_buf_idx_12;
                _ctx_l0ab_idx_iter_8 = _ctx_l0ab_idx_12;
                _ctx_l0c_idx_iter_8 = _ctx_l0c_idx_12;
                _ctx_task_id_iter_8 = _ctx_task_id_12;
            }

            auto _ctx_q_count_5 = (_ctx_q_count_iter_3 + 1);
            _ctx_buf_idx_iter_3 = _ctx_buf_idx_iter_8;
            _ctx_l0ab_idx_iter_3 = _ctx_l0ab_idx_iter_8;
            _ctx_l0c_idx_iter_3 = _ctx_l0c_idx_iter_8;
            _ctx_q_count_iter_3 = _ctx_q_count_5;
            _ctx_row_off_iter_3 = _ctx_row_off_5;
            _ctx_task_id_iter_3 = _ctx_task_id_iter_8;
        }

        _ctx_buf_idx_iter_1 = _ctx_buf_idx_iter_3;
        _ctx_l0ab_idx_iter_1 = _ctx_l0ab_idx_iter_3;
        _ctx_l0c_idx_iter_1 = _ctx_l0c_idx_iter_3;
        _ctx_q_count_iter_1 = _ctx_q_count_iter_3;
        _ctx_row_off_iter_1 = _ctx_row_off_iter_3;
        _ctx_sq_off_iter_1 = _ctx_sq_off_3;
        _ctx_task_id_iter_1 = _ctx_task_id_iter_3;
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    #endif  // __DAV_CUBE__
    #if defined(__DAV_VEC__)
    auto q_count_0 = 0;
    // Loop-carried values initialization
    auto q_count_iter_1 = q_count_0;

    for (uint64_t qi_1 = core_id_0; qi_1 < sq_tiles_0; qi_1 += num_cores_0) {
        auto sq_off_0 = (qi_1 * 128);
        // Loop-carried values initialization
        auto q_count_iter_3 = q_count_iter_1;

        for (uint64_t row_idx_1 = 0; row_idx_1 < 2; row_idx_1 += 1) {
            auto row_off_0 = (row_idx_1 * 64);
            auto q_idx_0 = (q_count_iter_3 % 2);
            using _tidx_52_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
            _tidx_52_0Type _tidx_52_0(64, 1);

            if ((q_idx_0 == 0)) {
                TASSIGN(_tidx_52_0, 0x14100);
            } else {
                TASSIGN(_tidx_52_0, 0x14200);
            }

            using _tidx_53_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
            _tidx_53_0Type _tidx_53_0(1, 64);

            if ((q_idx_0 == 0)) {
                TASSIGN(_tidx_53_0, 0x14100);
            } else {
                TASSIGN(_tidx_53_0, 0x14200);
            }

            using _tidx_54_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
            _tidx_54_0Type _tidx_54_0(64, 1);

            if ((q_idx_0 == 0)) {
                TASSIGN(_tidx_54_0, 0x14300);
            } else {
                TASSIGN(_tidx_54_0, 0x14400);
            }

            using _tidx_55_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
            _tidx_55_0Type _tidx_55_0(1, 64);

            if ((q_idx_0 == 0)) {
                TASSIGN(_tidx_55_0, 0x14300);
            } else {
                TASSIGN(_tidx_55_0, 0x14400);
            }

            for (uint64_t pre_1 = 0; pre_1 < 1; pre_1 += 1) {
                auto p_fifo_slot_0 = (pre_1 % 2);
                int64_t _tidx_56_0;

                if ((p_fifo_slot_0 == 0)) {
                    _tidx_56_0 = 0;
                } else {
                    _tidx_56_0 = 1;
                }

                if (_tidx_56_0 == 0) wait_flag_dev(0);
                if (_tidx_56_0 == 1) wait_flag_dev(1);
                if (_tidx_56_0 == 2) wait_flag_dev(2);
                auto p_fifo_slot_1 = (pre_1 % 2);
                auto skv_off_2 = (pre_1 * 128);
                TASSIGN(qk_buf_0Global, qk_buf_0 + ((((p_fifo_slot_1 * sq_dim_0) + sq_off_0) + row_off_0) * Skv + skv_off_2));
                TLOAD(qk_vec_0, qk_buf_0Global);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if ((pre_1 == 0)) {
                    TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                    TMULS(_tidx_53_0, reduce_dst_rm_0, 1.000000);
                    TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                    TEXP(qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TMULS(_tidx_55_0, reduce_dst_rm_0, 1.000000);
                    TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                }

                if ((pre_1 > 0)) {
                    TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TMAX(reduce_dst_rm_0, reduce_dst_rm_0, _tidx_53_0);
                    pipe_barrier(PIPE_V);
                    using _tidx_57_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_57_0Type _tidx_57_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_57_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_57_0, 0x14600);
                    }

                    TSUB(_tidx_57_0, _tidx_53_0, reduce_dst_rm_0);
                    pipe_barrier(PIPE_V);
                    TMULS(_tidx_53_0, reduce_dst_rm_0, 1.000000);
                    pipe_barrier(PIPE_V);
                    TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                    using _tidx_58_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_58_0Type _tidx_58_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_58_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_58_0, 0x14600);
                    }

                    using _tidx_59_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_59_0Type _tidx_59_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_59_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_59_0, 0x14600);
                    }

                    TMULS(_tidx_58_0, _tidx_59_0, 0.088388);
                    TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                    using _tidx_60_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_60_0Type _tidx_60_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_60_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_60_0, 0x14600);
                    }

                    using _tidx_61_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_61_0Type _tidx_61_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_61_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_61_0, 0x14600);
                    }

                    TEXP(_tidx_60_0, _tidx_61_0);
                    TEXP(qk_vec_0, tmp_vec_0);
                    TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                    pipe_barrier(PIPE_V);
                    using _tidx_62_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_62_0Type _tidx_62_0(1, 64);

                    if ((p_fifo_slot_1 == 0)) {
                        TASSIGN(_tidx_62_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_62_0, 0x14600);
                    }

                    TMUL(_tidx_55_0, _tidx_55_0, _tidx_62_0);
                    TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                    pipe_barrier(PIPE_V);
                    TADD(_tidx_55_0, _tidx_55_0, reduce_dst_rm_0);
                }

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TASSIGN(p_buf_0Global, p_buf_0 + ((((p_fifo_slot_1 * sq_dim_0) + sq_off_0) + row_off_0) * Skv + skv_off_2));
                TSTORE(p_buf_0Global, p_f16_0);
                int64_t _tidx_63_0;

                if ((p_fifo_slot_1 == 0)) {
                    _tidx_63_0 = 2;
                } else {
                    _tidx_63_0 = 3;
                }

                if (_tidx_63_0 == 0) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 0));
                if (_tidx_63_0 == 1) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 1));
                if (_tidx_63_0 == 2) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 2));
                if (_tidx_63_0 == 3) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 3));
                if (_tidx_63_0 == 4) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 4));
            }

            for (uint64_t ki_1 = 0; ki_1 < skv_tiles_0; ki_1 += 1) {
                auto next_ki_1 = (ki_1 + 1);
                if ((next_ki_1 < skv_tiles_0)) {
                    auto p_fifo_slot_2 = (next_ki_1 % 2);
                    int64_t _tidx_64_0;

                    if ((p_fifo_slot_2 == 0)) {
                        _tidx_64_0 = 0;
                    } else {
                        _tidx_64_0 = 1;
                    }

                    if (_tidx_64_0 == 0) wait_flag_dev(0);
                    if (_tidx_64_0 == 1) wait_flag_dev(1);
                    if (_tidx_64_0 == 2) wait_flag_dev(2);
                    auto p_fifo_slot_3 = (next_ki_1 % 2);
                    auto skv_off_3 = (next_ki_1 * 128);
                    TASSIGN(qk_buf_0Global, qk_buf_0 + ((((p_fifo_slot_3 * sq_dim_0) + sq_off_0) + row_off_0) * Skv + skv_off_3));
                    TLOAD(qk_vec_0, qk_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    if ((next_ki_1 == 0)) {
                        TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                        TMULS(_tidx_53_0, reduce_dst_rm_0, 1.000000);
                        TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                        TEXP(qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TMULS(_tidx_55_0, reduce_dst_rm_0, 1.000000);
                        TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                    }

                    if ((next_ki_1 > 0)) {
                        TROWMAX(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TMAX(reduce_dst_rm_0, reduce_dst_rm_0, _tidx_53_0);
                        pipe_barrier(PIPE_V);
                        using _tidx_65_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_65_0Type _tidx_65_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_65_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_65_0, 0x14600);
                        }

                        TSUB(_tidx_65_0, _tidx_53_0, reduce_dst_rm_0);
                        pipe_barrier(PIPE_V);
                        TMULS(_tidx_53_0, reduce_dst_rm_0, 1.000000);
                        pipe_barrier(PIPE_V);
                        TROWEXPANDSUB(tmp_vec_0, qk_vec_0, reduce_dst_0);
                        using _tidx_66_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_66_0Type _tidx_66_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_66_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_66_0, 0x14600);
                        }

                        using _tidx_67_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_67_0Type _tidx_67_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_67_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_67_0, 0x14600);
                        }

                        TMULS(_tidx_66_0, _tidx_67_0, 0.088388);
                        TMULS(tmp_vec_0, tmp_vec_0, 0.088388);
                        using _tidx_68_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_68_0Type _tidx_68_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_68_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_68_0, 0x14600);
                        }

                        using _tidx_69_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_69_0Type _tidx_69_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_69_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_69_0, 0x14600);
                        }

                        TEXP(_tidx_68_0, _tidx_69_0);
                        TEXP(qk_vec_0, tmp_vec_0);
                        TCVT(p_f16_0, qk_vec_0, RoundMode::CAST_ROUND);
                        pipe_barrier(PIPE_V);
                        using _tidx_70_0Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
                        _tidx_70_0Type _tidx_70_0(1, 64);

                        if ((p_fifo_slot_3 == 0)) {
                            TASSIGN(_tidx_70_0, 0x14500);
                        } else {
                            TASSIGN(_tidx_70_0, 0x14600);
                        }

                        TMUL(_tidx_55_0, _tidx_55_0, _tidx_70_0);
                        TROWSUM(reduce_dst_0, qk_vec_0, tmp_vec_0);
                        pipe_barrier(PIPE_V);
                        TADD(_tidx_55_0, _tidx_55_0, reduce_dst_rm_0);
                    }

                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TASSIGN(p_buf_0Global, p_buf_0 + ((((p_fifo_slot_3 * sq_dim_0) + sq_off_0) + row_off_0) * Skv + skv_off_3));
                    TSTORE(p_buf_0Global, p_f16_0);
                    int64_t _tidx_71_0;

                    if ((p_fifo_slot_3 == 0)) {
                        _tidx_71_0 = 2;
                    } else {
                        _tidx_71_0 = 3;
                    }

                    if (_tidx_71_0 == 0) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 0));
                    if (_tidx_71_0 == 1) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 1));
                    if (_tidx_71_0 == 2) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 2));
                    if (_tidx_71_0 == 3) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 3));
                    if (_tidx_71_0 == 4) ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, 4));
                }

                auto q_mat_idx_3 = (q_count_iter_3 % 2);
                auto pv_slot_0 = (ki_1 % 2);
                int64_t _tidx_72_0;

                if ((pv_slot_0 == 0)) {
                    _tidx_72_0 = 4;
                } else {
                    _tidx_72_0 = 5;
                }

                if (_tidx_72_0 == 0) wait_flag_dev(0);
                if (_tidx_72_0 == 1) wait_flag_dev(1);
                if (_tidx_72_0 == 2) wait_flag_dev(2);
                if (_tidx_72_0 == 3) wait_flag_dev(3);
                if (_tidx_72_0 == 4) wait_flag_dev(4);
                if (_tidx_72_0 == 5) wait_flag_dev(5);
                if (_tidx_72_0 == 6) wait_flag_dev(6);
                if ((ki_1 == 0)) {
                    TASSIGN(pv_buf_0Global, pv_buf_0 + (((((core_id_0 * 512) + ((q_mat_idx_3 * 2) * 128)) + (pv_slot_0 * 128)) + row_off_0) * D + 0));
                    TLOAD(running_o_0, pv_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                }

                if ((ki_1 > 0)) {
                    auto gu_fifo_slot_0 = (ki_1 % 2);
                    TASSIGN(pv_buf_0Global, pv_buf_0 + (((((core_id_0 * 512) + ((q_mat_idx_3 * 2) * 128)) + (pv_slot_0 * 128)) + row_off_0) * D + 0));
                    TLOAD(pv_vec_0, pv_buf_0Global);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    using _tidx_73_0Type = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
                    _tidx_73_0Type _tidx_73_0(64, 1);

                    if ((gu_fifo_slot_0 == 0)) {
                        TASSIGN(_tidx_73_0, 0x14500);
                    } else {
                        TASSIGN(_tidx_73_0, 0x14600);
                    }

                    TROWEXPANDMUL(running_o_0, running_o_0, _tidx_73_0);
                    TADD(running_o_0, running_o_0, pv_vec_0);
                }

            }

            auto q_count_5 = (q_count_iter_3 + 1);
            TROWEXPANDDIV(running_o_0, running_o_0, _tidx_54_0);
            TCVT(o_f16_0, running_o_0, RoundMode::CAST_ROUND);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TASSIGN(o_0Global, o_0 + ((sq_off_0 + row_off_0) * D + 0));
            TSTORE(o_0Global, o_f16_0);
            q_count_iter_3 = q_count_5;
        }

        q_count_iter_1 = q_count_iter_3;
    }

    #endif  // __DAV_VEC__
}
