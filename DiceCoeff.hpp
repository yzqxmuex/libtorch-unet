#include<torch/script.h>
#include <torch/torch.h>

//struct DiceCoeff : public torch::autograd::Function<DiceCoeff>
//{
//	float forward(torch::Tensor input, torch::Tensor target)
//	{
//		float eps = 0.0001;
//		torch::Tensor inter = torch::dot(input.view(-1), target.view(-1));
//		torch::Tensor union_ = torch::sum(input) + torch::sum(target) + eps;
//		float t = (2 * inter.item().toFloat() + eps) / union_.item().toFloat();
//		
//		return t;
//	}
//};

float dice_coeff(torch::Tensor input, torch::Tensor target)
{
	float eps = 0.0001;
	torch::Tensor inter = torch::dot(input.view(-1), target.view(-1));
	torch::Tensor union_ = torch::sum(input) + torch::sum(target);
	float t = (2 * inter.item().toFloat() + eps) / union_.item().toFloat() + eps;

	return t;
}