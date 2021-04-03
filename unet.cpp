// unet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<torch/script.h>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/parallel/data_parallel.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstddef>
#include "Carvana_Load.hpp"
#include "args.hxx"
#include "option.hpp"
#include "unet.hpp"
#include "DiceCoeff.hpp"

using namespace torch;
using namespace cv;

namespace F = torch::nn::functional;

auto ToCvImage(at::Tensor tensor)
{
	int width = tensor.sizes()[1];
	int height = tensor.sizes()[0];
	try
	{
		cv::Mat output_mat(cv::Size{ width, height }, CV_8UC3, tensor.data_ptr<uchar>());
		return output_mat.clone();
	}
	catch (const c10::Error& e)
	{
		std::cout << "an error has occured : " << e.msg() << std::endl;
	}
	return cv::Mat(height, width, CV_8UC3);
}

torch::DeviceType device_type;

struct FloatReader
{
	void operator()(const std::string &name, const std::string &value, std::tuple<double, double> &destination)
	{
		size_t commapos = 0;
		std::get<0>(destination) = std::stod(value, &commapos);
		std::get<1>(destination) = std::stod(std::string(value, commapos + 1));
	}
};

COptionMap<int> COptInt;
COptionMap<bool> COptBool;
COptionMap<std::string> COptString;
COptionMap<double> COptDouble;
COptionMap<std::tuple<double, double>> COptTuple;

class Train
{
public:
	Train() {	}
	~Train() {	}

public:
	template <typename DataLoader>
	void train(int nEpoch,
				UNET& unet,
				torch::Device device,
				DataLoader& data_loader,
				torch::optim::Optimizer& optimizer,
				torch::nn::BCEWithLogitsLoss& BCEWithLogits_criterion,
				torch::nn::CrossEntropyLoss& CrossEntryopy_criterion,
				int n_class);

	template <typename DataLoader>
	void val(int nEpoch, UNET& unet, torch::Device device, DataLoader& data_loader, int n_class);
};

template <typename DataLoader>
void Train::train(int nEpoch,
					UNET& unet,
					torch::Device device,
					DataLoader& train_loader,
					torch::optim::Optimizer& optimizer,
					torch::nn::BCEWithLogitsLoss& BCEWithLogits_criterion,
					torch::nn::CrossEntropyLoss& CrossEntryopy_criterion,
					int n_class)
{
	unet.to(device);
	unet.train(true);

	int global_step = 0;
	long long accumulationCost = 0;
	for (auto batch : *train_loader)
	{
		auto data = batch.data();
		torch::Tensor imgs = data->data;
		torch::Tensor true_masks = data->target;
		//cout << imgs.sizes() << endl;
		//imgs = imgs.squeeze(0);
		//imgs = imgs.to(torch::kCPU);
		//imgs = imgs.mul_(255);
		//imgs = imgs.toType(torch::kByte);
		//true_masks = true_masks.squeeze(0);
		//imgs = imgs.permute({ 1, 2, 0 }).contiguous();
		//cv::Mat imgsMat = ToCvImage(imgs);
		////cv::Mat maskMat = ToCvImage(true_masks);
		//cv::imshow("imgsMat", imgsMat);
		////cv::imshow("maskMat", maskMat);
		//waitKey(5000);
		imgs = imgs.to(device);
		true_masks = true_masks.to(device);

		optimizer.zero_grad();
		auto tm_start = std::chrono::system_clock::now();
		torch::Tensor masks_pred = unet.forward(imgs);
		auto tm_end = std::chrono::system_clock::now();

		if (n_class > 1)
		{
			auto loss1 = CrossEntryopy_criterion(masks_pred, true_masks);
			loss1.backward();
			torch::nn::utils::clip_grad_value_(unet.parameters(), 0.1);
			optimizer.step();
			accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
			global_step += 1;
			if (global_step % 458 == 0)
			{
				printf("epoch {%d}, global_step{%d}, loss: {%.3f} cost:{%lld msec}\n",
					nEpoch,
					global_step,
					loss1.item().toFloat(),
					accumulationCost);
				accumulationCost = 0;
			}
		}
		else
		{
			auto loss2 = BCEWithLogits_criterion(masks_pred, true_masks);
			loss2.backward();
			torch::nn::utils::clip_grad_value_(unet.parameters(), 0.1);
			optimizer.step();
			accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
			global_step += 1;
			if (global_step % 458 == 0)
			{
				printf("epoch {%d}, global_step{%d}, loss: {%.3f} cost:{%lld msec}\n",
					nEpoch,
					global_step,
					loss2.item().toFloat(),
					accumulationCost);
				accumulationCost = 0;
			}
		}
	}
}

template <typename DataLoader>
void Train::val(int nEpoch, UNET& unet, torch::Device device, DataLoader& valid_loader, int n_class)
{
	{
		torch::NoGradGuard no_grad_guard;
		unet.to(device);
		unet.train(false);
		unet.eval();

		float tot = 0;
		
		for (auto batch : *valid_loader)
		{
			auto data = batch.data();
			torch::Tensor imgs = data->data;
			torch::Tensor true_masks = data->target;
			imgs = imgs.to(device);
			true_masks = true_masks.to(/*device*/torch::kCPU);
			auto tm_start = std::chrono::system_clock::now();
			torch::Tensor masks_pred = unet.forward(imgs);
			auto tm_end = std::chrono::system_clock::now();
			if (n_class > 1)
			{
				tot += F::cross_entropy(masks_pred, true_masks).item().toFloat();
			}
			else
			{
				torch::Tensor pred = torch::sigmoid(masks_pred);
				pred = pred.to(torch::kCPU);
				float* temp_arr = (float*)pred.data_ptr();
				//sigmoid求出每个像素的概率值
				//将大于0.5的和小于0.5的分别转换成1, 0
				for (int i = 0; i < pred.sizes()[2]; i ++)
				{
					for (int j = 0; j < pred.sizes()[3]; j++)
					{
						if (((*temp_arr) - 0.5) > 0)
						{
							(*temp_arr++) = 1;
						}
						else
						{
							(*temp_arr++) = 0;
						}
					}
				}
				tot = dice_coeff(pred, true_masks);
				printf("epoch {%d}, dice_coeff: {%.3f} \n", nEpoch, tot);
			}
		}
		
	}
}

int main(int argc, char **argv)
{
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    
	srand(time(NULL));
	args::ArgumentParser parser("Train the UNet on images and target masks.", "This goes after the options.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<int> n_class(parser, "n_class", "Pixel classification", { "n_class" }, 1);
	args::ValueFlag<int> n_channels(parser, "n_channels", "RGB images", { "n_channels" }, 3);
	args::ValueFlag<bool> isbilinear(parser, "bilinear", "if net.bilinear else Transposed conv", { "bilinear" }, 1);
	args::ValueFlag<int> epochs(parser, "epochs", "number of epochs to train", { "epochs" }, 5);
	args::ValueFlag<int> batch_size(parser, "batch_size", "train patch size", { "batch_size" }, 1);
	args::ValueFlag<double> lr(parser, "lr", "learning rate", { "lr" }, 1e-4);
	args::ValueFlag<float> scale(parser, "scale", "Downscaling factor of the images", { "scale" }, 0.5);
	args::ValueFlag<std::string> str_imgs_train_dir(parser, "imgs_train_dir", "train dataset directory", { "imgs_train_dir" }, "..\\Carvana\\imgs_train");
	args::ValueFlag<std::string> str_dir_train_mask(parser, "dir_train_mask", "label dataset directory", { "dir_train_mask" }, "..\\Carvana\\masks_train");
	args::ValueFlag<std::string> str_imgs_valid_dir(parser, "imgs_valid_dir", "valid dataset directory", { "imgs_valid_dir" }, "..\\Carvana\\imgs_valid");
	args::ValueFlag<std::string> str_dir_valid_mask(parser, "dir_valid_mask", "label valid dataset directory", { "dir_valid_mask" }, "..\\Carvana\\masks_valid");

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}
	catch (args::ValidationError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	int Epochs = (int)args::get(epochs);
	int batch = (int)args::get(batch_size);
	double learning_rate = (double)args::get(lr);
	float scale_factor = (float)args::get(scale);
	
	int nclass = (int)args::get(n_class);
	int channels = (int)args::get(n_channels);
	bool bilinear = (bool)args::get(isbilinear);
	string imgs_dir_train = (string)args::get(str_imgs_train_dir);
	string dir_mask_train = (string)args::get(str_dir_train_mask);
	string imgs_dir_valid = (string)args::get(str_imgs_valid_dir);
	string dir_mask_valid = (string)args::get(str_dir_valid_mask);

	cout << "Epochs : " << Epochs << endl;
	cout << "batch : " << batch << endl;
	cout << "learning_rate : " << learning_rate << endl;
	cout << "scale : " << scale_factor << endl;
	
	cout << "n_channels : " << channels << endl;
	cout << "n_class : " << nclass << endl;
	cout << "bilinear : " << bilinear << endl;
	cout << "imgs_train : " << imgs_dir_train << endl;
	cout << "mask_train : " << dir_mask_train << endl;
	cout << "imgs_valid : " << imgs_dir_valid << endl;
	cout << "mask_valid : " << dir_mask_valid << endl;
	
	std::vector<double> norm_mean = { 0.485, 0.456, 0.406 };
	std::vector<double> norm_std = { 0.229, 0.224, 0.225 };
	auto train_dataset = CarvanaDataset(imgs_dir_train, dir_mask_train, scale_factor)/*.map(torch::data::transforms::Normalize<>(norm_mean, norm_std))*/;
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batch);

	auto valid_dataset = CarvanaDataset(imgs_dir_valid, dir_mask_valid, scale_factor);
	auto valid_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(valid_dataset), batch);

	Train T;
	UNET unet(channels, nclass, bilinear);
	//cout << unet << endl;

	torch::nn::BCEWithLogitsLoss BCEWithLogits_criterion;
	torch::nn::CrossEntropyLoss CrossEntryopy_criterion;
	if (nclass > 1)
	{
		CrossEntryopy_criterion = torch::nn::CrossEntropyLoss();
		CrossEntryopy_criterion->to(device);
	}
	else
	{
		BCEWithLogits_criterion = torch::nn::BCEWithLogitsLoss();
		BCEWithLogits_criterion->to(device);
	}
	
	auto optimizer = torch::optim::RMSprop(unet.parameters(), torch::optim::RMSpropOptions(learning_rate).momentum(0.9).weight_decay(1e-8));

	for (int e = 0; e < Epochs; e ++)
	{
		if (((e + 1) % 4) == 0)
		{
			learning_rate *= 0.5;
			static_cast<torch::optim::RMSpropOptions &>(optimizer.param_groups()[0].options()).lr(learning_rate);
		}
		for (auto param_group : optimizer.param_groups()) {

			printf("lr = %.9f ", static_cast<torch::optim::RMSpropOptions &>(param_group.options()).lr());
		}
		T.train(e, unet, device, train_loader, optimizer, BCEWithLogits_criterion, CrossEntryopy_criterion, nclass);
		T.val(e, unet, device, valid_loader, nclass);
	}

	printf("Finish training!\n");
	torch::serialize::OutputArchive archive;
	unet.save(archive);
	archive.save_to("..\\retModel\\unet.pt");
	printf("Save the training result to ..\\unet.pt.\n");
	return 0;
}
