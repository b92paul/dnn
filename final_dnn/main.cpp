#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/final/f_1943.lab";
char trainPath[] = "../../data/final/f_train.ark";
char testPath[]  = "../../data/final/f_test.ark";
char testId[]    = "../../data/final/test_id.out";
#define INPUT_SIZE 69
#define NN_INPUT_SIZE 483
#define OUTPUT_SIZE 1943
#define TRAIN_READ 1124823 // Max is 1124823
#define TEST_READ 166114 // Max is 166114
#define MOM 0.0
#define ETA 0.1
#define BATCH_NUM 50000000 // 1 epoch about 1e6 data
#define BATCH_SIZE 128
#define VAL_SIZE 24823
#define TIME_DECAY true
#define TIME_DECAY_NUM 500000.0
#define NORM 7
/*
mt19937 rng(0x5EED);
int randint(int lb, int ub) {
		return uniform_int_distribution<int>(lb, ub)(rng);
}
*/
typedef vector<double> VD;
typedef vector<VectorXd> VXd;
int check = 100000;
#define DONE(x) printf("%s done",x)
void csvToMatrix(char* filename,  MatrixXd& out,int length, int cut, double norm = 1){
	int idx = 0;
	out = MatrixXd(length, cut);
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	double num;char split;
	if(csv == NULL){
		puts("no such file !!!");
    assert(false);
	}
	double tmp;
  for(int i= 0;i<cut;i++){
    for(int j = 0;j<length;j++){
      if(~fscanf(csv,"%lf",&tmp)){
        out(j,i) = tmp/norm;
      } else{
        printf("%d %d\n",i,j);
      }
    }
    if((i+1)%200000==0)printf("read data: %d\n",i+1);
  }
  puts("done");
	fclose(csv);
  //DONE(filename);
}
void matrixExpansion(MatrixXd& A){
	printf("%lu %lu\n",A.rows(),A.cols());
  int copy = NN_INPUT_SIZE/INPUT_SIZE;
	MatrixXd tmp(A.rows()* copy, A.cols());
	int len = A.rows();
	int total = A.cols();
	for(int i=0; i<total;i++){
		for(int j=0 ;j< copy;j++){
			int idx = (i - (copy-1) + j*2+total) % total;
			tmp.block(j*len,i, len, 1) = A.col(idx);
		}	
	}
	A = tmp;
	printf("%lu %lu\n",A.rows(),A.cols());
}
void shuffleMatrix(MatrixXd& X,MatrixXd& Y, MatrixXd& vX, MatrixXd& vY, int val_size){
	puts("-- In shuffleMatrix");
	int n = X.cols();
	if(val_size > 0 ){
		vX = MatrixXd( NN_INPUT_SIZE, val_size);
		vY = MatrixXd(1, val_size);
	}
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
void labelToMatrix(char* filename, MatrixXd& out, int length, int cut){
	int idx = 0;
	out = MatrixXd::Zero(1, cut);
  puts("read label!!");
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	if(csv == NULL){
		puts("no such file !!!");
	}

	int tmp;
  for(int i = 0; i < cut;i++){
    fscanf(csv,"%d",&tmp);
    out(0,i) = tmp;
  }	
 
	fclose(csv);
}
int main(){
	MatrixXd inputX, inputY;
	MatrixXd valX, valY;
	csvToMatrix(trainPath, inputX, INPUT_SIZE, TRAIN_READ, NORM);
	labelToMatrix(labelPath, inputY,OUTPUT_SIZE, TRAIN_READ);
	matrixExpansion(inputX);
	shuffleMatrix(inputX, inputY, valX, valY, VAL_SIZE);
	printf("X size = %lu\n",inputX.rows());
	printf("Y size = %lu\n",inputY.rows());
	printf("input X data size = %lu\n",inputX.cols());
	printf("input Y data size = %lu\n",inputY.cols());
	printf("val X data size = %lu\n",valX.cols());
	printf("val Y data size = %lu\n",valY.cols());
  for(int i=0;i<10;i++){
    printf("%d\n",(int)inputY(0,i));
  }
	// new network
	vector<int> layer;
	layer.push_back(768);
	layer.push_back(768);
	//layer.push_back(150);
	//layer.push_back(150);
	//layer.push_back(150);
	layer.push_back(OUTPUT_SIZE);	
	NetWork nn(layer, NN_INPUT_SIZE, MOM, true);
	nn.outsize = OUTPUT_SIZE;
	MatrixXd testX;
	nn.SGD(inputX, inputY, ETA, BATCH_NUM, BATCH_SIZE, TIME_DECAY, TIME_DECAY_NUM, valX, valY, testX);
	
	//read test data
  /*
	csvToMatrix(testPath, testX, INPUT_SIZE, TEST_READ, NORM);
	matrixExpansion(testX);
	printf("test data size = %lu\n",testX.cols());
  */
	//nn.Predict(testX);
	
	return 0;

}
