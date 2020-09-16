#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <THC/THCGeneral.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>
#include <cmath>
#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

typedef enum{
  ADAM_MODE_0   =0, // eps under square root
  ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

template <int DEPTH, typename T, typename GRAD_T>
struct DistAdamFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<DEPTH>& tl,
    const float* per_tensor_beta1,
    const float* per_tensor_beta2,
    const int* per_tensor_bias_correction,
    const float* per_tensor_eps,
    const float* per_tensor_weight_decay,
    const float lr,
    const float grad_scale,
    const int step,
    adamMode_t mode)
  {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float b1 = per_tensor_beta1[tensor_num];
    float b2 = per_tensor_beta2[tensor_num];
    float eps = per_tensor_eps[tensor_num];
    float decay = per_tensor_weight_decay[tensor_num];

    float bias_correction1, bias_correction2, step_size;
    if (per_tensor_bias_correction[tensor_num] == 1) {
      bias_correction1 = 1 - std::pow(b1, step);
      bias_correction2 = 1 - std::pow(b2, step);
      step_size = lr * std::sqrt(bias_correction2)/bias_correction1;
    } else {
      step_size = lr;
    }

    T* p = (T *)tl.addresses[0][tensor_loc];
    p += chunk_idx*chunk_size;
    T* m = (T *)tl.addresses[1][tensor_loc];
    m += chunk_idx*chunk_size;
    T* v = (T *)tl.addresses[2][tensor_loc];
    v += chunk_idx*chunk_size;
    GRAD_T* g = (GRAD_T *)tl.addresses[3][tensor_loc];
    g += chunk_idx*chunk_size;
    GRAD_T* p_copy = NULL;
    if (DEPTH == 5) {
      p_copy = (GRAD_T *)tl.addresses[4][tensor_loc];
      p_copy += chunk_idx*chunk_size;
    }

    n -= chunk_idx*chunk_size;
    
    T incoming_p[ILP];
    T incoming_m[ILP];
    T incoming_v[ILP];
    T incoming_g[ILP];

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 &&
      chunk_size % ILP == 0 &&
      is_aligned(p) &&
      is_aligned(m) &&
      is_aligned(v) &&
      is_aligned(g) &&
      is_aligned(p_copy)) {
      for (int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x) {
        // load
        GRAD_T tmp_g[ILP];
        load_store(incoming_p, p, 0, i_start);
        load_store(incoming_m, m, 0, i_start);
        load_store(incoming_v, v, 0, i_start);
        load_store(tmp_g, g, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          T old_p = incoming_p[ii];
	  T old_m = incoming_m[ii];
	  T old_v = incoming_v[ii];
	  incoming_g[ii] = static_cast<T>(tmp_g[ii]);
          T scaled_grad = incoming_g[ii]/grad_scale;
          incoming_m[ii] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
          incoming_v[ii] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
          float denom;
          if (mode == ADAM_MODE_0)
            denom = sqrtf(incoming_v[ii] + eps);
          else // Mode 1
            denom = sqrtf(incoming_v[ii]) + eps;
          float update = (incoming_m[ii]/denom) + (decay*incoming_p[ii]);
          incoming_p[ii] = incoming_p[ii] - (step_size*update);
	  if (tensor_loc==1 && i_start == 0 && ii == 0) {
            printf("tensor_loc:%d,tensor_num:%d,g:%.16f,grad_scale:%f,scaled_grad:%.16f,old_p:%.16f,old_m:%.8f,old_v:%.8f,b1:%.8f,b2:%.8f,b1c:%.8f,b2c:%.8f,new_m:%.16f,new_v:%.16f,eps:%f,denom:%.16f,decay:%f,update:%.16f,step_size:%f,lr:%f,new_p:%.16f\n", tensor_loc, tensor_num, incoming_g[ii], grad_scale, scaled_grad, old_p, old_m, old_v, b1, b2, bias_correction1, bias_correction1, incoming_m[ii], incoming_v[ii], eps, denom, decay, update, step_size, lr, incoming_p[ii]);
          }
          if (DEPTH == 5)  tmp_g[ii] = static_cast<GRAD_T>(incoming_p[ii]);
        }
        load_store(p, incoming_p, i_start, 0);
        load_store(m, incoming_m, i_start, 0);
        load_store(v, incoming_v, i_start, 0);
        if (DEPTH == 5) load_store(p_copy, tmp_g, i_start, 0);
      }
    } else {
      for (int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP) {

#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          incoming_p[ii] = 0;
          incoming_m[ii] = 0;
          incoming_v[ii] = 0;
          incoming_g[ii] = 0;

          int i = i_start + threadIdx.x + ii*blockDim.x;
          if (i < n && i < chunk_size) {
            incoming_p[ii] = p[i];
            incoming_m[ii] = m[i];
            incoming_v[ii] = v[i];
            incoming_g[ii] = static_cast<T>(g[i]);
          }
        }

#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int j = i_start + threadIdx.x + ii*blockDim.x;

          if (j < n && j < chunk_size) {
            T scaled_grad = incoming_g[ii]/grad_scale;
            m[j] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
            v[j] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
            float denom;
            if (mode == ADAM_MODE_0)
              denom = sqrtf(v[j] + eps);
            else // Mode 1
              denom = sqrtf(v[j]) + eps;
            float update = (m[j]/denom) + (decay*incoming_p[ii]);
            p[j] = incoming_p[ii] - (step_size*update);
            if (DEPTH == 5)  p_copy[j] = (GRAD_T) p[j];
          }
        }
      }
    }
  }
};

void multi_tensor_fused_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,  // p, m, v, g, p_copy
  at::Tensor per_tensor_beta1,
  at::Tensor per_tensor_beta2,
  at::Tensor per_tensor_bias_correction,
  at::Tensor per_tensor_eps,
  at::Tensor per_tensor_weight_decay,
  float lr,
  float grad_scale,
  int step,
  int mode)
{
  using namespace at;

  size_t tl_sz = tensor_lists.size();
  AT_ASSERTM(tl_sz == 4 || tl_sz == 5, "expected tensor lists of size 4 or 5");

  if (tl_sz == 5) {
    DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 0, "dist_adam_cuda_kernel",  // g
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      multi_tensor_apply<5>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        DistAdamFunctor<5, accscalar_t, scalar_t_0>(),
        per_tensor_beta1.DATA_PTR<float>(),
        per_tensor_beta2.DATA_PTR<float>(),
        per_tensor_bias_correction.DATA_PTR<int>(),
        per_tensor_eps.DATA_PTR<float>(),
        per_tensor_weight_decay.DATA_PTR<float>(),
        lr,
        grad_scale,
        step,
        (adamMode_t) mode);
    );
  } else {
    DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 0, "dist_adam_cuda_kernel",  // g
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        DistAdamFunctor<4, accscalar_t, scalar_t_0>(),
        per_tensor_beta1.DATA_PTR<float>(),
        per_tensor_beta2.DATA_PTR<float>(),
        per_tensor_bias_correction.DATA_PTR<int>(),
        per_tensor_eps.DATA_PTR<float>(),
        per_tensor_weight_decay.DATA_PTR<float>(),
        lr,
        grad_scale,
        step,
        (adamMode_t) mode);
    );
  }
  THCudaCheck(cudaGetLastError());
}
