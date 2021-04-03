// inference.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
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

using namespace torch;
using namespace cv;
namespace F = torch::nn::functional;

auto ToCvImage(at::Tensor tensor)
{
	int width = tensor.sizes()[1];
	int height = tensor.sizes()[0];
	try
	{
		cv::Mat output_mat(cv::Size{ width, height }, CV_8UC1, tensor.data_ptr<uchar>());
		return output_mat.clone();
	}
	catch (const c10::Error& e)
	{
		std::cout << "an error has occured : " << e.msg() << std::endl;
	}
	return cv::Mat(height, width, CV_8UC1);
}

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
			auto tm_start = std::chrono::system_clock::now();
			torch::Tensor masks_pred = unet.forward(imgs);
			auto tm_end = std::chrono::system_clock::now();

			masks_pred = masks_pred.to(torch::kCPU);
			masks_pred = masks_pred.squeeze(0);
			masks_pred = masks_pred.toType(torch::kByte);
			masks_pred = masks_pred.permute({1, 2, 0}).contiguous();
			cv::Mat mat = ToCvImage(masks_pred);
			namedWindow("Display window_pred", CV_WINDOW_AUTOSIZE);
			imshow("Display window_pred", mat);

			true_masks = true_masks.to(torch::kCPU);
			true_masks = true_masks.mul(255);
			true_masks = true_masks.squeeze(0);
			true_masks = true_masks.toType(torch::kByte);
			true_masks = true_masks.permute({ 1, 2, 0 }).contiguous();
			cv::Mat mat_true = ToCvImage(true_masks);
			namedWindow("Display window_true", CV_WINDOW_AUTOSIZE);
			imshow("Display window_true", mat_true);
			waitKey(5000);
		}

	}
}

int main(int argc, char **argv)
{
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Running on GPU." : "Training on CPU.") << '\n';

	srand(time(NULL));
	args::ArgumentParser parser("Train the UNet on images and target masks.", "This goes after the options.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<int> n_class(parser, "n_class", "Pixel classification", { "n_class" }, 1);
	args::ValueFlag<int> n_channels(parser, "n_channels", "RGB images", { "n_channels" }, 3);
	args::ValueFlag<bool> isbilinear(parser, "bilinear", "if net.bilinear else Transposed conv", { "bilinear" }, 1);
	args::ValueFlag<float> scale(parser, "scale", "Downscaling factor of the images", { "scale" }, 0.5);
	args::ValueFlag<int> batch_size(parser, "batch_size", "train patch size", { "batch_size" }, 1);
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

	int batch = (int)args::get(batch_size);
	float scale_factor = (float)args::get(scale);
	string imgs_dir_valid = (string)args::get(str_imgs_valid_dir);
	string dir_mask_valid = (string)args::get(str_dir_valid_mask);
	int nclass = (int)args::get(n_class);
	int channels = (int)args::get(n_channels);
	bool bilinear = (bool)args::get(isbilinear);


	auto valid_dataset = CarvanaDataset(imgs_dir_valid, dir_mask_valid, scale_factor);
	auto valid_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(valid_dataset), batch);

	Train T;
	UNET unet(channels, nclass, bilinear);
	torch::serialize::InputArchive archive;
	archive.load_from("..\\retModel\\unet.pt");

	unet.load(archive);
	T.val(0, unet, device, valid_loader, nclass);

	printf("Finish infer!\n");
}
