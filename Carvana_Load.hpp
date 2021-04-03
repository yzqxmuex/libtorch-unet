#include<torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>        // for strcpy(), strcat()
#include <io.h>
#include <stdio.h>
#include <random>
#include <math.h>

using namespace std;
using namespace torch;
using namespace cv;

typedef struct stuFileInfoList
{
	std::string		strFilePath;
	std::string		strFileName;
}STUFILEINFOLIST, *PSTUFILEINFOLIST;

typedef std::list< STUFILEINFOLIST > fileNameList_t;

void append(fileNameList_t &List, STUFILEINFOLIST &info);
void for_each(fileNameList_t &List);

void append(fileNameList_t &List, STUFILEINFOLIST &info)
{
	List.push_back(info);
}

int getListSize(fileNameList_t List)
{
	return List.size();
}

void for_each(fileNameList_t &List)
{
	fileNameList_t::iterator iter;
	for (iter = List.begin(); iter != List.end(); iter++)
	{
		std::cout << iter->strFilePath << std::endl;
		std::cout << iter->strFileName << std::endl;
	}
}

string getfile(fileNameList_t &List, int idx)
{
	string s = "";
	fileNameList_t::iterator iter;
	int i = 0;
	for (iter = List.begin(); iter != List.end(); iter++, i ++)
	{
		if (i == idx)
			return iter->strFilePath;
	}
	return s;
}

class CarvanaDataset : public torch::data::Dataset<CarvanaDataset>
{
public:
	explicit CarvanaDataset(string imgs_dir, string masks_dir, float scale = 1);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

	fileNameList_t		idsImg;
	fileNameList_t		idsMask;
	std::string			strExt;
private:
	string imgs_dir, masks_dir;
	float scale;
	void _scan_(std::string strDirdata, std::string strExt, fileNameList_t& _t);
	void _listFiles_(const char * dir, fileNameList_t&);
	torch::Tensor preprocess_imgs(cv::Mat, float scale);
	torch::Tensor preprocess_marks(cv::Mat, float scale);
};

CarvanaDataset::CarvanaDataset(string imgs_dir, string masks_dir, float scale):scale(scale)
{
	strExt = "jpg";
	_scan_(imgs_dir, strExt, idsImg);
	_scan_(masks_dir, strExt, idsMask);
}

void CarvanaDataset::_scan_(std::string strDirdata, std::string strExt, fileNameList_t& _t)
{
	_listFiles_(strDirdata.c_str(), _t);
	//for_each(_t);
}

void CarvanaDataset::_listFiles_(const char * dir, fileNameList_t& _t)
{
	std::string		strDir = dir;
	char dirNew[200];
	strcpy_s(dirNew, 200, dir);
	strcat_s(dirNew, 200, "\\*.");
	strcat_s(dirNew, 200, strExt.c_str());
	//cout << "dirNew : " << dirNew << endl;

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // 检查是否成功
		return;

	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			strcpy_s(dirNew, 200, dir);
			strcat_s(dirNew, 200, "\\");
			strcat_s(dirNew, 200, findData.name);

			_listFiles_(dirNew, _t);
		}
		else
		{
			STUFILEINFOLIST stuFileInfo;
			stuFileInfo.strFilePath = strDir + "\\" + findData.name;
			stuFileInfo.strFileName = findData.name;
			append(_t, stuFileInfo);
		}
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // 关闭搜索句柄
}

torch::Tensor CarvanaDataset::preprocess_imgs(cv::Mat mat, float scale)
{
	int w = mat.cols;
	int h = mat.rows;
	
	int newW = int(scale * w);
	int newH = int(scale * h);

	if (newW < 0 || newH < 0)
	{
		cout << "Scale is too small" << endl;
		torch::Tensor a;
		return a;
	}
	cv::Mat newMat(newH, newW, CV_8UC3);
	cv::Size dsize(newW, newH);
	cv::resize(mat, newMat, dsize);
	//cv::imshow("newMat", newMat);
	//waitKey(5000);

	torch::Tensor img_tensor = torch::from_blob(newMat.data, { newMat.rows, newMat.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous(); // Channels x Height x Width
	img_tensor = img_tensor.toType(torch::kFloat32);
	img_tensor = img_tensor.unsqueeze(0);
	img_tensor = img_tensor.div_(255);
	return img_tensor.clone();
}

torch::Tensor CarvanaDataset::preprocess_marks(cv::Mat mat, float scale)
{
	int w = mat.cols;
	int h = mat.rows;

	int newW = int(scale * w);
	int newH = int(scale * h);

	if (newW < 0 || newH < 0)
	{
		cout << "Scale is too small" << endl;
		torch::Tensor a;
		return a;
	}
	cv::Mat newMat(newH, newW, CV_8UC1);
	cv::Size dsize(newW, newH);
	cv::resize(mat, newMat, dsize);
	//cv::imshow("newMat", newMat);
	//waitKey(5000);

	torch::Tensor img_tensor = torch::from_blob(newMat.data, { newMat.rows, newMat.cols, 1 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous(); // Channels x Height x Width
	img_tensor = img_tensor.toType(torch::kFloat32);
	img_tensor = img_tensor.unsqueeze(0);
	img_tensor = img_tensor.div_(255);		//一定要在float这个类型下归一化数据,如果在kByte下归一化模型是无法收敛的
	return img_tensor.clone();
}

torch::data::Example<> CarvanaDataset::get(size_t index)
{
	torch::Tensor img, mask;

	string imgname = getfile(idsImg, index);
	cv::Mat imgMat = cv::imread(imgname);
	img = preprocess_imgs(imgMat, scale);

	string maskname = getfile(idsMask, index);
	cv::Mat maskMat = cv::imread(maskname, IMREAD_GRAYSCALE);
	mask = preprocess_marks(maskMat, scale);

	return { img.clone(), mask.clone() };
}

torch::optional<size_t> CarvanaDataset::size() const
{
	return getListSize(idsImg);
}