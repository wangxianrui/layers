#include <vector>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe
{

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    // Trim layer input
    if (this->phase_ == TEST)
    {
        for (int i = 0; i < bottom.size(); ++i)
        {
            this->QuantizeLayerInputs_gpu(bottom[i]->mutable_gpu_data(), bottom[i]->count());
        }
    }

    // forward
    const Dtype *bottom_data = bottom[0]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    int num = bottom[0]->shape(0);
    int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0) * this->channels_);

    if (bottom[0] != top[0])
    {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    if (this->use_global_stats_)
    {
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ? 0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_gpu_scale(this->variance_.count(), scale_factor, this->blobs_[0]->gpu_data(), this->mean_.mutable_gpu_data());
        caffe_gpu_scale(this->variance_.count(), scale_factor, this->blobs_[1]->gpu_data(), this->variance_.mutable_gpu_data());

        // trim mean and variance  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        this->QuantizeWeights_gpu(this->blobs_, rounding, true);
    }
    else
    {
        // compute mean
        caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1. / (num * spatial_dim), bottom_data, this->spatial_sum_multiplier_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0., this->mean_.mutable_gpu_data());
        /*
        // trim mean  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        int cnt = this->mean_.count();
        int bit_width = this->bw_layer_in_;
        int fl = this->fl_layer_in_;
        //Dtype *data = this->mean_.mutable_gpu_data();
        for (int index = 0; index < cnt; ++index) {
            // Saturate data
            Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
            Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
            this->mean_.mutable_gpu_data()[index] = std::max(std::min(this->mean_.mutable_gpu_data()[index], max_data), min_data);
            // Round data
            this->mean_.mutable_gpu_data()[index] /= pow(2, -fl);
            switch (rounding) {
            case QuantizationParameter_Rounding_NEAREST:
                this->mean_.mutable_gpu_data()[index] = round(this->mean_.mutable_gpu_data()[index]);
                break;
            case QuantizationParameter_Rounding_STOCHASTIC:
                this->mean_.mutable_gpu_data()[index] = floor(this->mean_.mutable_gpu_data()[index] + rand() / (RAND_MAX+1.0));
                break;
            default:
                break;
            }
            this->mean_.mutable_gpu_data()[index] *= pow(2, -fl);
        }
        */
    }

    // subtract mean
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, -1, this->num_by_chans_.gpu_data(), this->spatial_sum_multiplier_.gpu_data(), 1., top_data);

    if (!this->use_global_stats_)
    {
        // compute variance using var(X) = E((X-EX)^2)
        caffe_powx(top[0]->count(), top_data, Dtype(2), this->temp_.mutable_gpu_data()); // (X-EX)^2
        caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1. / (num * spatial_dim), this->temp_.gpu_data(), this->spatial_sum_multiplier_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0., this->variance_.mutable_gpu_data()); // E((X_EX)^2)
        /*
        // trim variance  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        int cnt = this->variance_.count();
        int bit_width = this->bw_layer_in_;
        int fl = this->fl_layer_in_;
        //Dtype *data = this->variance_.mutable_gpu_data();
        for (int index = 0; index < cnt; ++index) {
            // Saturate data
            Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
            Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
            this->variance_.mutable_gpu_data()[index] = std::max(std::min(this->variance_.mutable_gpu_data()[index], max_data), min_data);
            // Round data
            this->variance_.mutable_gpu_data()[index] /= pow(2, -fl);
            switch (rounding) {
            case QuantizationParameter_Rounding_NEAREST:
                this->variance_.mutable_gpu_data()[index] = round(this->variance_.mutable_gpu_data()[index]);
                break;
            case QuantizationParameter_Rounding_STOCHASTIC:
                this->variance_.mutable_gpu_data()[index] = floor(this->variance_.mutable_gpu_data()[index] + rand() / (RAND_MAX+1.0));
                break;
            default:
                break;
            }
            this->variance_.mutable_gpu_data()[index] *= pow(2, -fl);
        }
        */

        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= this->moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_gpu_axpby(this->mean_.count(), Dtype(1), this->mean_.gpu_data(), this->moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
        int m = bottom[0]->count() / this->channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
        caffe_gpu_axpby(this->variance_.count(), bias_correction_factor, this->variance_.gpu_data(), this->moving_average_fraction_, this->blobs_[1]->mutable_gpu_data());
    }

    // normalize variance
    caffe_gpu_add_scalar(this->variance_.count(), this->eps_, this->variance_.mutable_gpu_data());
    caffe_gpu_powx(this->variance_.count(), this->variance_.gpu_data(), Dtype(0.5), this->variance_.mutable_gpu_data());

    // replicate variance to input size
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.gpu_data(), this->variance_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, 1., this->num_by_chans_.gpu_data(), this->spatial_sum_multiplier_.gpu_data(), 0., this->temp_.mutable_gpu_data());
    caffe_div(this->temp_.count(), top_data, this->temp_.gpu_data(), top_data);
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    caffe_copy(this->x_norm_.count(), top_data, this->x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
    const Dtype *top_diff;
    if (bottom[0] != top[0])
    {
        top_diff = top[0]->gpu_diff();
    }
    else
    {
        caffe_copy(this->x_norm_.count(), top[0]->gpu_diff(), this->x_norm_.mutable_gpu_diff());
        top_diff = this->x_norm_.gpu_diff();
    }
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->use_global_stats_)
    {
        caffe_gpu_div(this->temp_.count(), top_diff, this->temp_.gpu_data(), bottom_diff);
        return;
    }
    const Dtype *top_data = this->x_norm_.gpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0) * this->channels_);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,     // respectively, dE/dY is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.

    // sum(dE/dY \cdot Y)
    caffe_gpu_mul(this->temp_.count(), top_data, top_diff, bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1., bottom_diff, this->spatial_sum_multiplier_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0., this->mean_.mutable_gpu_data());

    // reshape (broadcast) the above
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, 1., this->num_by_chans_.gpu_data(), this->spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_mul(this->temp_.count(), top_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1., top_diff, this->spatial_sum_multiplier_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0., this->mean_.mutable_gpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0., this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * this->channels_, spatial_dim, 1, 1., this->num_by_chans_.gpu_data(), this->spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_gpu_axpby(this->temp_.count(), Dtype(1), top_diff, Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // note: this->temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_gpu_div(this->temp_.count(), bottom_diff, this->temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormRistrettoLayer);

} // namespace caffe