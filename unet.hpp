#include<torch/script.h>
#include <torch/torch.h>
#include"unet_parts.hpp"

struct UNET : public torch::nn::Module
{
	UNET(int n_channels = 3, int n_classes = 1, bool bilinear = true):n_channels(n_channels), n_classes(n_classes), bilinear_(bilinear)
	{
		inc = register_module("inc", std::make_shared<DoubleConv>(n_channels, 64));
		down1 = register_module("down1", std::make_shared<Down>(64, 128));
		down2 = register_module("down2", std::make_shared<Down>(128, 256));
		down3 = register_module("down3", std::make_shared<Down>(256, 512));
		if (bilinear_) factor = 2; else factor = 1;
		down4 = register_module("down4", std::make_shared<Down>(512, 1024 / factor));
		up1 = register_module<Up>("up1", std::make_shared<Up>(1024, 512 / factor, bilinear_));
		up2 = register_module<Up>("up2", std::make_shared<Up>(512, 256 / factor, bilinear));
		up3 = register_module<Up>("up3", std::make_shared<Up>(256, 128 / factor, bilinear));
		up4 = register_module<Up>("up4", std::make_shared<Up>(128, 64, bilinear));
		outc = register_module<OutConv>("outc", std::make_shared<OutConv>(64, n_classes));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x1 = inc(x);
		x2 = down1(x1);
		x3 = down2(x2);
		x4 = down3(x3);
		x5 = down4(x4);
		x = up1(x5, x4);
		x = up2(x, x3);
		x = up3(x, x2);
		x = up4(x, x1);
		logits = outc(x);
		return logits;
	}

	torch::nn::ModuleHolder<DoubleConv>	inc{ nullptr };
	torch::nn::ModuleHolder<Down> down1{ nullptr }, down2{ nullptr }, down3{ nullptr }, down4{ nullptr };
	torch::nn::ModuleHolder<Up>	up1{ nullptr }, up2{ nullptr }, up3{ nullptr }, up4{ nullptr };
	torch::nn::ModuleHolder<OutConv> outc{ nullptr };

	torch::Tensor x1,x2,x3,x4,x5;
	torch::Tensor logits;
private:
	int n_channels;
	int n_classes;
	bool bilinear_;
	int factor;
};