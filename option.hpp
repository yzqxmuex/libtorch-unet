#ifndef _OPTION_H
#define	_OPTION_H

#include <iostream>

using namespace std;

template <class G_Type>
class COptionMap
{
public:
	COptionMap();
	~COptionMap();

	void InsertOptMap(std::string strOpt, const G_Type item);
	void GetOptMap(std::string strOpt, G_Type &item);

private:
	std::map<std::string, G_Type> optmap_;
};

template <class G_Type>
COptionMap<G_Type>::COptionMap()
{

}

template <class G_Type>
COptionMap<G_Type>::~COptionMap()
{

}

template <class G_Type>
void COptionMap<G_Type>::InsertOptMap(std::string strOpt, const G_Type item)
{
	pair<std::string, G_Type>abc(strOpt, item);
	optmap_.insert(abc);
}

template <class G_Type>
void COptionMap<G_Type>::GetOptMap(std::string strOpt, G_Type &item)
{
	std::map<std::string, G_Type>::iterator		it;

	it = optmap_.find(strOpt);
	if (it != optmap_.end())
	{
		item = it->second;
		//cout << it->second << endl;
	}
}

#endif // !_OPTION_H
