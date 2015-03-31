#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "fast_dnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label_39.out";
char trainPath[] = "../../data/merge/train.out";
char testPath[]  = "../../data/merge/test.out";
char testId[]    = "../../data/merge/test_id.out";
#define INPUT_SIZE 108
#define OUTPUT_SIZE 39 
#define TRAIN_READ 1124823 // Max is 1124823
#define TEST_READ 180406 // Max is 180406
#define MOM 0.9
#define ETA 0.4
#define BATCH_NUM 50000000
#define BATCH_SIZE 500
#define VAL_SIZE 10000

mt19937 rng(0x5EED);
int randint(int lb, int ub) {
		return uniform_int_distribution<int>(lb, ub)(rng);
}

typedef vector<double> VD;
typedef vector<VectorXd> VXd;
int check = 100000;
void csvToMatrix(char* filename,  MatrixXd& out,int length, int cut){
	int idx = 0;
	out = MatrixXd(length, cut);
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	double num;char split;
	if(csv == NULL){
		puts("no such file !!!");
	}
	VectorXd tmpXd;
	vector<double> tmp;
	while(~fscanf(csv,"%lf%c",&num, &split)){
		tmp.push_back(num);
		if(split=='\n'){
			tmpXd = VectorXd::Map(&tmp[0],tmp.size());
			tmp.clear();
			out.col(idx++) << tmpXd;
			if((idx) % check ==0)printf("read data: %d\n",idx);
			// break point
			if(idx == cut) { fclose(csv); return;}
		}
	}
	if(!tmp.empty()){
		tmpXd = VectorXd::Map(&tmp[0],tmp.size());
		out.col(idx++) << tmpXd;
	}
	if(cut > out.cols()) out.conservativeResize(Eigen::NoChange, cut);
	fclose(csv);
}
void shuffleMatrix(MatrixXd& X,MatrixXd& Y, MatrixXd& vX, MatrixXd& vY, int val_size){
	puts("-- In shuffleMatrix");
	int n = X.cols();
	if(val_size > 0 ){
		vX = MatrixXd( INPUT_SIZE, val_size);
		vY = MatrixXd(OUTPUT_SIZE, val_size);
	}
	MatrixXd tmpX, tmpY;
	for(int i = n-1; i >= 0; i--){
		int idx = randint(0,i);
		X.col(i).swap(X.col(idx));
		Y.col(i).swap(Y.col(idx));
		if(n - i - 1 < val_size){
			vX.col(n-i-1) << X.col(i);
			vY.col(n-i-1) << Y.col(i);
		}
	}
	X.conservativeResize(Eigen::NoChange, X.cols()- val_size);
	Y.conservativeResize(Eigen::NoChange, Y.cols()- val_size);
	puts("-- Done shuffleMatrix");
}

int main(){
	MatrixXd inputX, inputY;
	MatrixXd valX, valY;
	csvToMatrix(trainPath, inputX, INPUT_SIZE, TRAIN_READ);
	csvToMatrix(labelPath, inputY,OUTPUT_SIZE, TRAIN_READ);
	shuffleMatrix(inputX, inputY, valX, valY, VAL_SIZE);
	printf("X size = %lu\n",inputX.rows());
	printf("Y size = %lu\n",inputY.rows());
	printf("input X data size = %lu\n",inputX.cols());
	printf("input Y data size = %lu\n",inputY.cols());
	printf("val X data size = %lu\n",valX.cols());
	printf("val Y data size = %lu\n",valY.cols());

	// new network
	vector<int> layer;
	layer.push_back(128);
	//layer.push_back(100);
	//layer.push_back(100);
	//layer.push_back(100);
	//layer.push_back(100);
	layer.push_back(OUTPUT_SIZE);	
	NetWork nn(layer, INPUT_SIZE, MOM, true);
	nn.outsize = OUTPUT_SIZE;
	
	//read test data
	MatrixXd testX;
	csvToMatrix(testPath, testX, INPUT_SIZE, TEST_READ);
	printf("test data size = %lu\n",testX.cols());
	nn.SGD(inputX, inputY, ETA, BATCH_NUM, BATCH_SIZE, valX, valY, testX);

	//nn.Predict(testX);
	
	return 0;

}
