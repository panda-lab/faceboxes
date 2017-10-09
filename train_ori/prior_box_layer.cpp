#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/prior_box_layer.hpp"
#include <iostream>
using namespace std;
namespace caffe {

template <typename Dtype>
void PriorBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	const PriorBoxParameter& prior_box_param =
			this->layer_param_.prior_box_param();
	CHECK_GT(prior_box_param.min_size_size(), 0) << "must provide min_size.";
	for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
		min_sizes_.push_back(prior_box_param.min_size(i));
		CHECK_GT(min_sizes_.back(), 0) << "min_size must be positive.";
	}
	aspect_ratios_.clear();
	aspect_ratios_.push_back(1.);
	if (min_sizes_.size() == 3)
		num_priors_ = aspect_ratios_.size() * 21;
	else
		num_priors_ = aspect_ratios_.size();

	if (prior_box_param.variance_size() > 1) {
		// Must and only provide 4 variance.
		CHECK_EQ(prior_box_param.variance_size(), 4);
		for (int i = 0; i < prior_box_param.variance_size(); ++i) {
			CHECK_GT(prior_box_param.variance(i), 0);
			variance_.push_back(prior_box_param.variance(i));
		}
	} else if (prior_box_param.variance_size() == 1) {
		CHECK_GT(prior_box_param.variance(0), 0);
		variance_.push_back(prior_box_param.variance(0));
	} else {
		// Set default to 0.1.
		variance_.push_back(0.1);
	}
	offset_ = prior_box_param.offset();
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	const int layer_width = bottom[0]->width();
	const int layer_height = bottom[0]->height();
	vector<int> top_shape(3, 1);
	// Since all images in a batch has same height and width, we only need to
	// generate one set of priors which can be shared across all images.
	top_shape[0] = 1;
	// 2 channels. First channel stores the mean of each prior coordinate.
	// Second channel stores the variance of each prior coordinate.
	top_shape[1] = 2;
	top_shape[2] = layer_width * layer_height * num_priors_ * 4;
	CHECK_GT(top_shape[2], 0);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	const int layer_width = bottom[0]->width();
	const int layer_height = bottom[0]->height();

	int img_width, img_height;
	img_width = bottom[1]->width();
	img_height = bottom[1]->height();

	Dtype* top_data = top[0]->mutable_cpu_data();
	int dim = layer_height * layer_width * num_priors_ * 4;
	int idx = 0;



	for(int s = 0; s < min_sizes_.size(); ++s)
	{
		int min_size_ = min_sizes_[s];
		if(min_size_ == 32)
		{
			int step = min_size_ / 4;
			int num_w = img_width / step;
			int num_h = img_height / step;
			for (int h = 0; h < num_h; ++h)
			{
				for (int w = 0; w < num_w; ++w)
				{
					float center_x = (w + 0.5) * step;
					float center_y = (h + 0.5) * step;
					top_data[idx++] = (center_x - min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y - min_size_ / 2.) / img_height;
					top_data[idx++] = (center_x + min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y + min_size_ / 2.) / img_height;
				}
			}
		}
		if(min_size_ == 64)
		{
			int step = min_size_ / 2;
			int num_w = img_width / step;
			int num_h = img_height / step;
			for (int h = 0; h < num_h; ++h)
			{
				for (int w = 0; w < num_w; ++w)
				{
					float center_x = (w + 0.5) * step;
					float center_y = (h + 0.5) * step;
					top_data[idx++] = (center_x - min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y - min_size_ / 2.) / img_height;
					top_data[idx++] = (center_x + min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y + min_size_ / 2.) / img_height;
				}
			}
		}
		else
		{
			int step = min_size_;
			int num_w = img_width / step;
			int num_h = img_height / step;
			for (int h = 0; h < num_h; ++h)
			{
				for (int w = 0; w < num_w; ++w)
				{
					float center_x = (w + 0.5) * step;
					float center_y = (h + 0.5) * step;
					top_data[idx++] = (center_x - min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y - min_size_ / 2.) / img_height;
					top_data[idx++] = (center_x + min_size_ / 2.) / img_width;
					top_data[idx++] = (center_y + min_size_ / 2.) / img_height;
				}
			}
		}
	}


	top_data += top[0]->offset(0, 1);
	if (variance_.size() == 1) {
		caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
	} else {
		int count = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				for (int i = 0; i < num_priors_; ++i) {
					for (int j = 0; j < 4; ++j) {
						top_data[count] = variance_[j];
						++count;
					}
				}
			}
		}
	}
}

INSTANTIATE_CLASS(PriorBoxLayer);
REGISTER_LAYER_CLASS(PriorBox);

}	// namespace caffe
