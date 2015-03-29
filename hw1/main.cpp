#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn_fast.cpp"
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
	fclose(csv);
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

	// norm input vector
	double vnorm = 1;

	VXd valX,valY,trainX,trainY;
	for(int i=0;i<val_size;i++){
		valX.push_back(inputX[idx[i]]/vnorm);
		valY.push_back(inputY[idx[i]]/vnorm);
	}
	for(int i = val_size; i<inputX.size();i++){
		trainX.push_back(inputX[idx[i]]/vnorm);
		trainY.push_back(inputY[idx[i]]/vnorm);
	}

	// new network
	vector<int> layer;
	layer.push_back(200);
	layer.push_back(100);
	layer.push_back(60);
	layer.push_back(inputY[0].size());

	int input_size = inputX[0].size();
	NetWork nn(layer,input_size,0.9,true);
	
	//read test data
	vector<VectorXd> testX = csvToVecters(testPath);
	printf("data size = %lu\n",testX.size());
	nn.SGD(trainX,trainY,0.4 ,300000,500,valX,valY,testX);

	//nn.Predict(testX);
	
	return 0;

}
