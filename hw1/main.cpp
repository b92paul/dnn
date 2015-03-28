#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label.out";
char trainPath[] = "../../data/merge/train.out";
char testPath[]  = "../../data/merge/test.out";
char testId[]    = "../../data/merge/test_id.out";

typedef vector<double> VD;
typedef vector<VectorXd> VXd;
int check = 100000;
vector<VectorXd> csvToVecters(char* filename, int cut=300000){
	int idx =0 ;
	vector<VectorXd> res;
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	double num;
	char split;
	vector<double> tmp;
	if(csv == NULL){
		puts("no such file !!!");
		return res;
	}
	VectorXd tmpXd;
	while(~fscanf(csv,"%lf%c",&num, &split)){
		tmp.push_back(num);
		if(split=='\n'){
			tmpXd = VectorXd::Map(&tmp[0],tmp.size());
			res.push_back(tmpXd);
			tmp.clear();
			if(res.size()%check ==0)printf("read data: %lu\n",res.size());
			idx++;
			if(cut==idx)return res;
		}	
	}
	if(!tmp.empty()){
		tmpXd = VectorXd::Map(&tmp[0],tmp.size());
		res.push_back(tmpXd);
	}
	return res;
}

int main(){
	vector<int> idx;
	vector<VectorXd> inputX = csvToVecters(trainPath);
	printf("data size = %lu\n",inputX.size());
	vector<VectorXd> inputY = csvToVecters(labelPath);
	printf("data size = %lu\n",inputY.size());
	int val_size = 5000;
	printf("X size = %lu\n",inputX[0].size());
	printf("Y size = %lu\n",inputY[0].size());
	for(int i=0;i<inputX.size();i++)idx.push_back(i);
	random_shuffle(idx.begin(),idx.end());
	
	VXd valX,valY,trainX,trainY;
	for(int i=0;i<val_size;i++){
		valX.push_back(inputX[idx[i]]);
		valY.push_back(inputY[idx[i]]);
	}
	for(int i = val_size; i<inputX.size();i++){
		trainX.push_back(inputX[idx[i]]);
		trainY.push_back(inputY[idx[i]]);
	}
/*

	VXd valX = VXd(inputX.begin(),inputX.begin()+val_size);
	VXd valY = VXd(inputY.begin(),inputY.begin()+val_size);
	VXd trainX = VXd(inputX.begin()+val_size,inputX.end());
	VXd trainY = VXd(inputY.begin()+val_size,inputY.end());
*/

	vector<int> layer;
	layer.push_back(28);
	layer.push_back(58);
	layer.push_back(inputY[0].size());
	int input_size = inputX[0].size();
	NetWork nn(layer,input_size);
	nn.SGD(trainX,trainY,1,1000,500,valX,valY);
	


	/* try to out put	
	for(int i=0;i<res.size();i++){
		for(int j=0;j<res[i].size();j++){
			printf("%lf,",res[i][j]);
		}
		puts("");
	}
	*/
	return 0;

}
