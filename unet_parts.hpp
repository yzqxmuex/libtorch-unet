#include<torch/script.h>
#include <torch/torch.h>

//try
//{
//	double_conv->forward(input);
//}
//catch (const c10::Error& e)
//{
//	std::cerr << e.msg();
//}
namespace F = torch::nn::functional;

//(convolution => [BN] => ReLU) * 2
struct DoubleConv : public torch::nn::Module
{
	DoubleConv(int in_channels, int out_channels, int mid_channels = 0)
	{
		if (mid_channels == 0)
			mid_channels = out_channels;

		torch::nn::Sequential doubleconv(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, mid_channels, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(mid_channels)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, out_channels, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)));

		double_conv = register_module("double_conv", doubleconv.ptr());
		
	}
	torch::Tensor forward(torch::Tensor input)
	{
		return double_conv->forward(input);
	}

	torch::nn::Sequential double_conv{ nullptr };
};
//Downscaling with maxpool then double conv
struct Down : public torch::nn::Module
{
	Down(int in_channels, int out_channels)
	{
		torch::nn::Sequential maxpoolconv(
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
			DoubleConv(in_channels, out_channels));
		maxpool_conv = register_module("maxpool_conv", maxpoolconv.ptr());
	}

	torch::Tensor forward(torch::Tensor input)
	{
		return maxpool_conv->forward(input);
	}

	torch::nn::Sequential maxpool_conv{ nullptr };
};


//Р§зг:F::pad(input, F::PadFuncOptions({ 1, 2, 2, 1, 1, 2 }).mode(torch::kReplicate));

struct Up : public torch::nn::Module
{
	Up(int in_channels, int out_channels, bool bilinear = true):isBilinear(bilinear)
	{
		if (bilinear)
		{
			up_Upsample = register_module("up", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kBilinear).align_corners(true)));
			conv = register_module("conv", std::make_shared<DoubleConv>(in_channels, out_channels, in_channels / 2));
		}
		else
		{
			up_ConvTranspose2d = register_module("up", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_channels, in_channels / 2, 2).stride(2)));
			conv = register_module("conv", std::make_shared<DoubleConv>(in_channels, out_channels));
		}
	}

	torch::Tensor forward(torch::Tensor x1, torch::Tensor x2)
	{
		if (isBilinear)
			x1 = up_Upsample(x1);
		else
			x1 = up_ConvTranspose2d(x1);
		//# input is CHW
		diffY = x2.sizes()[2] - x1.sizes()[2];
		diffX = x2.sizes()[3] - x1.sizes()[3];
		x1 = F::pad(x1, F::PadFuncOptions({ diffX / 2, diffX - diffX / 2, diffY / 2, diffY - diffY / 2 }));
		x = torch::cat({x2, x1}, 1);
	
		return conv->forward(x);
	}

	torch::nn::Upsample			up_Upsample{ nullptr };
	torch::nn::ModuleHolder<DoubleConv>	conv{ nullptr };
	torch::nn::ConvTranspose2d	up_ConvTranspose2d{ nullptr };
	bool	isBilinear;
	int		diffY, diffX;
	torch::Tensor	x;
};

struct OutConv : public torch::nn::Module
{
	OutConv(int in_channels, int out_channels)
	{
		kernel_size = 1;
		conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return conv(x);
	}

	torch::nn::Conv2d conv{ nullptr };
	int kernel_size;
};