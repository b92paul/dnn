#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn_fast.cpp"
#include <random>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cstring>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label.out";
char trainPath[] = "../../data/merge/f_train.out";
char testPath[]  = "../../data/merge/f_test.out";
char testId[]    = "../../data/merge/test_id.out";

typedef vector<double> VD;
typedef vector<VectorXd> VXd;
int check = 100000;
vector<VectorXd> csvToVecters(char* filename, int cut=-1){
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
string itoa(int n){char buf[10];sprintf(buf,"%d",n);return buf;}
void go(vector<int>layer,double eta,double mom, VXd& trainX, VXd& trainY, VXd& valX, VXd& valY) {
	int input_size = trainX[0].size();
	NetWork nn(layer,input_size,mom);
	vector<int>params;
	vector<pair<double,double> > ret;
	params.push_back(1000);
	params.push_back(20000);
	VXd valTest;
	nn.SGD(trainX,trainY,eta ,100000,500,valX,valY,valTest,true,&params,&ret);
	char filename[100];
	string tmp = "";
	for(int i=0;i<layer.size()-1;i++) {
		if(i!=0)tmp += "_";
		tmp += itoa(layer[i]);
	}
	sprintf(filename, "models/model_finder[%s]_%d_%d.res",tmp.c_str(),(int)(eta*10),(int)(mom*10));
	FILE *f = fopen(filename, "w");
	for(int i=0;i<ret.size();i++)
		fprintf(f,"%lf %lf\n",ret[i].first,ret[i].second);
	//e_valid, e_in
	fclose(f);

}
int randint(int a,int b) {
	return rand()%(b-a+1)+a;
}
void model_finder(VXd& trainX, VXd& trainY, VXd& valX, VXd& valY) {
	for(int i=0;i<50;i++) {
		vector<int>layer;
		int layer_size = randint(3,4);
		while(layer_size--){
			layer.push_back(randint(5,15)*10);
		}
		layer.push_back(trainY[0].size());
		for(int k=2;k<=8;k+=2)
			for(int j=1;j<=5;j++) {
				double mom = 0.1*k;
				double eta = 0.1*(j);
				go(layer,eta,mom,trainX,trainY,valX,valY);
			}
	}
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
	srand(time(NULL));
	random_shuffle(idx.begin(),idx.end());
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

	model_finder(trainX,trainY,valX,valY);

	return 0;

}
