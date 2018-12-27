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
BatchNormRistrettoLayer<Dtype>::BatchNormRistrettoLayer(
    const LayerParameter &param) : BatchNormLayer<Dtype>(param), BaseRistrettoLayer<Dtype>()
{
    this->precision_ = this->layer_param_.quantization_param().precision();
    this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
    switch (this->precision_)
    {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
        this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
        this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
        this->bw_params_ = this->layer_param_.quantization_param().bw_params();
        this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
        this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
        this->fl_params_ = this->layer_param_.quantization_param().fl_params();
        break;
    case QuantizationParameter_Precision_MINIFLOAT:
        this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
        this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
        break;
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
        this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
        this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
        this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
        this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
        this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
        this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
        break;
    default:
        LOG(FATAL) << "Unknown precision mode: " << this->precision_;
        break;
    }
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    BatchNormParameter param = this->layer_param_.batch_norm_param();
    this->moving_average_fraction_ = param.moving_average_fraction();
    this->use_global_stats_ = this->phase_ == TEST;
    if (param.has_use_global_stats())
        this->use_global_stats_ = param.use_global_stats();
    if (bottom[0]->num_axes() == 1)
        this->channels_ = 1;
    else
        this->channels_ = bottom[0]->shape(1);
    this->eps_ = param.eps();
    if (this->blobs_.size() > 0)
    {
        LOG(INFO) << "Skipping parameter initialization";
    }
    else
    {
        this->blobs_.resize(3);
        vector<int> sz;
        sz.push_back(this->channels_);
        this->blobs_[0].reset(new Blob<Dtype>(sz));
        this->blobs_[1].reset(new Blob<Dtype>(sz));
        sz[0] = 1;
        this->blobs_[2].reset(new Blob<Dtype>(sz));
        for (int i = 0; i < 3; ++i)
        {
            caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_data());
        }
    }
    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.
    for (int i = 0; i < this->blobs_.size(); ++i)
    {
        if (this->layer_param_.param_size() == i)
        {
            ParamSpec *fixed_param_spec = this->layer_param_.add_param();
            fixed_param_spec->set_lr_mult(0.f);
        }
        else
        {
            CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
                << "Cannot configure batch normalization statistics as layer "
                << "parameters.";
        }
    }
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    // Trim layer input
    if (this->phase_ == TEST)
    {
        for (int i = 0; i < bottom.size(); ++i)
        {
            this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data(), bottom[i]->count());
        }
    }

    // forward
    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
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
        caffe_cpu_scale(this->variance_.count(), scale_factor, this->blobs_[0]->cpu_data(), this->mean_.mutable_cpu_data());
        caffe_cpu_scale(this->variance_.count(), scale_factor, this->blobs_[1]->cpu_data(), this->variance_.mutable_cpu_data());

        // trim mean and variance  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        this->QuantizeWeights_cpu(this->blobs_, rounding, true);
    }
    else
    {
        // compute mean
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1. / (num * spatial_dim), bottom_data, this->spatial_sum_multiplier_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.cpu_data(), this->batch_sum_multiplier_.cpu_data(), 0., this->mean_.mutable_cpu_data());
        /*
        // trim mean  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        int cnt = this->mean_.count();
        int bit_width = this->bw_layer_in_;
        int fl = this->fl_layer_in_;
        //Dtype *data = this->mean_.mutable_cpu_data();
        for (int index = 0; index < cnt; ++index) 
        {
            // Saturate data
            Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
            Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
            this->mean_.mutable_cpu_data()[index] = std::max(std::min(this->mean_.mutable_cpu_data()[index], max_data), min_data);
            // Round data
            this->mean_.mutable_cpu_data()[index] /= pow(2, -fl);
            switch (rounding) {
            case QuantizationParameter_Rounding_NEAREST:
                this->mean_.mutable_cpu_data()[index] = round(this->mean_.mutable_cpu_data()[index]);
                break;
            case QuantizationParameter_Rounding_STOCHASTIC:
                this->mean_.mutable_cpu_data()[index] = floor(this->mean_.mutable_cpu_data()[index] + rand() / (RAND_MAX+1.0));
                break;
            default:
                break;
            }
            this->mean_.mutable_cpu_data()[index] *= pow(2, -fl);
        }
        */
    }

    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.cpu_data(), this->mean_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, -1, this->num_by_chans_.cpu_data(), this->spatial_sum_multiplier_.cpu_data(), 1., top_data);

    if (!this->use_global_stats_)
    {
        // compute variance using var(X) = E((X-EX)^2)
        caffe_powx(top[0]->count(), top_data, Dtype(2), this->temp_.mutable_cpu_data()); // (X-EX)^2
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1. / (num * spatial_dim), this->temp_.cpu_data(), this->spatial_sum_multiplier_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.cpu_data(), this->batch_sum_multiplier_.cpu_data(), 0., this->variance_.mutable_cpu_data()); // E((X_EX)^2)
        /*
        // trim variance  ---- wxrui
        int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
        int cnt = this->variance_.count();
        int bit_width = this->bw_layer_in_;
        int fl = this->fl_layer_in_;
        //Dtype *data = this->variance_.mutable_cpu_data();
        for (int index = 0; index < cnt; ++index) {
            // Saturate data
            Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
            Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
            this->variance_.mutable_cpu_data()[index] = std::max(std::min(this->variance_.mutable_cpu_data()[index], max_data), min_data);
            // Round data
            this->variance_.mutable_cpu_data()[index] /= pow(2, -fl);
            switch (rounding) {
            case QuantizationParameter_Rounding_NEAREST:
                this->variance_.mutable_cpu_data()[index] = round(this->variance_.mutable_cpu_data()[index]);
                break;
            case QuantizationParameter_Rounding_STOCHASTIC:
                this->variance_.mutable_cpu_data()[index] = floor(this->variance_.mutable_cpu_data()[index] + rand() / (RAND_MAX+1.0));
                break;
            default:
                break;
            }
            this->variance_.mutable_cpu_data()[index] *= pow(2, -fl);
        }
        */

        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= this->moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_cpu_axpby(this->mean_.count(), Dtype(1), this->mean_.cpu_data(), this->moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count() / this->channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
        caffe_cpu_axpby(this->variance_.count(), bias_correction_factor, this->variance_.cpu_data(), this->moving_average_fraction_, this->blobs_[1]->mutable_cpu_data());
    }

    // normalize variance
    caffe_add_scalar(this->variance_.count(), this->eps_, this->variance_.mutable_cpu_data());
    caffe_powx(this->variance_.count(), this->variance_.cpu_data(), Dtype(0.5), this->variance_.mutable_cpu_data());

    // replicate variance to input size
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.cpu_data(), this->variance_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, 1., this->num_by_chans_.cpu_data(), this->spatial_sum_multiplier_.cpu_data(), 0., this->temp_.mutable_cpu_data());
    caffe_div(this->temp_.count(), top_data, this->temp_.cpu_data(), top_data);
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    caffe_copy(this->x_norm_.count(), top_data, this->x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
    const Dtype *top_diff;
    if (bottom[0] != top[0])
    {
        top_diff = top[0]->cpu_diff();
    }
    else
    {
        caffe_copy(this->x_norm_.count(), top[0]->cpu_diff(), this->x_norm_.mutable_cpu_diff());
        top_diff = this->x_norm_.cpu_diff();
    }
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->use_global_stats_)
    {
        caffe_div(this->temp_.count(), top_diff, this->temp_.cpu_data(), bottom_diff);
        return;
    }
    const Dtype *top_data = this->x_norm_.cpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0) * this->channels_);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,   // respectively, dE/dY is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.

    // sum(dE/dY \cdot Y)
    caffe_mul(this->temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1., bottom_diff, this->spatial_sum_multiplier_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.cpu_data(), this->batch_sum_multiplier_.cpu_data(), 0., this->mean_.mutable_cpu_data());

    // reshape (broadcast) the above
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.cpu_data(), this->mean_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num, spatial_dim, 1, 1., this->num_by_chans_.cpu_data(), this->spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_mul(this->temp_.count(), top_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1., top_diff, this->spatial_sum_multiplier_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1., this->num_by_chans_.cpu_data(), this->batch_sum_multiplier_.cpu_data(), 0., this->mean_.mutable_cpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1, this->batch_sum_multiplier_.cpu_data(), this->mean_.cpu_data(), 0., this->num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * this->channels_, spatial_dim, 1, 1., this->num_by_chans_.cpu_data(), this->spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_cpu_axpby(this->temp_.count(), Dtype(1), top_diff, Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // note: this->temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_div(this->temp_.count(), bottom_diff, this->temp_.cpu_data(), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(BatchNormRistrettoLayer);
#endif

INSTANTIATE_CLASS(BatchNormRistrettoLayer);
REGISTER_LAYER_CLASS(BatchNormRistretto);

} // namespace caffe
