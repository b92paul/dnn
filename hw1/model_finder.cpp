#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn_fast.cpp"
#include <random>
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
#include <algorithm>
void go(vector<int>layer,double eta,VXd& trainX, VXd& trainY, VXd& valX, VXd& valY) {
	int input_size = trainX[0].size();
	NetWork nn(layer,input_size);
	vector<int>params;
	vector<pair<double,double> > ret;
	params.push_back(100);
	params.push_back(2000);
	VXd valTest;
	nn.SGD(trainX,trainY,eta ,100000,500,valX,valY,valTest,true,&params,&ret);
	char filename[100];
	if(layer.size() < 3+1)layer[2]=0;
	sprintf(filename, "models/model_finder_%d_%d_%d_%d.res",layer[0],layer[1],layer[2],(int)(eta*10));
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
		int a=randint(2,10)*10;
		int b=randint(2,10)*10;
		layer.push_back(a);
		layer.push_back(b);
		if(rand()&1) {
			layer.push_back(randint(2,10)*10);
		}
		layer.push_back(trainY[0].size());
		for(int j=3;j<=7;j++) {
			double eta = 0.1*(j);
			go(layer,eta,trainX,trainY,valX,valY);
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
	random_shuffle(idx.begin(),idx.end());
srand(time(NULL));
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


	/* try to out put	
	for(int i=0;i<res.siz:w
	e();i++){
		for(int j=0;j<res[i].size();j++){
			printf("%lf,",res[i][j]);
		}
		puts("");
	}
	*/
	return 0;

}
