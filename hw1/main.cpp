#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn.cpp"
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

int main(){
	vector<VectorXd> inputX = csvToVecters(trainPath);
	printf("data size = %lu\n",inputX.size());
	vector<VectorXd> inputY = csvToVecters(labelPath);
	printf("data size = %lu\n",inputY.size());
	int val_size = 5000;
	VXd valX = VXd(inputX.begin(),inputX.begin()+val_size);
	VXd valY = VXd(inputY.begin(),inputY.begin()+val_size);
	VXd trainX = VXd(inputX.begin()+val_size,inputX.end());
	VXd trainY = VXd(inputY.begin()+val_size,inputY.end());


	vector<int> layer;
	layer.push_back(128);
	layer.push_back(inputY[0].size());
	VXd OAO;


	int input_size = inputX[0].size();
	NetWork nn(layer,input_size);
	nn.SGD(trainX,trainY,0.01,1000,2000,valX,valY);
	


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
