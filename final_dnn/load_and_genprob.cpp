#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "dnn.cpp"
#include <random>
#include <cassert>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label_392.out";
char trainPath[] = "../../data/final/f_train.ark";
char testPath[]  = "../..//data/final/f_test_hw1.ark";
char testId[]    = "../../data/merge/test_id2.out";
#define INPUT_SIZE 69
#define NN_INPUT_SIZE 621
#define OUTPUT_SIZE 1943 
#define TRAIN_READ 112482 // Max is 1124823
#define TEST_READ 180406 // Max is 180406 166114
#define MOM 0.0
#define ETA 0.1
#define BATCH_NUM 50000000 // 1 epoch about 1e6 data
#define BATCH_SIZE 128
#define VAL_SIZE 24823
#define TIME_DECAY true
#define TIME_DECAY_NUM 500000.0
#define NORM 7

char model_name[] = "model_0.5312_0.4448.out";
/*
mt19937 rng(0x5EED);
int randint(int lb, int ub) {
		return uniform_int_distribution<int>(lb, ub)(rng);
}
*/
char buf[10000];
typedef vector<double> VD;
typedef vector<VectorXd> VXd;
vector<int> word_size;
vector<string> word_name;
int check = 100000;
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



int main(){

	// new network
	vector<int> layer;
	layer.push_back(512);
	layer.push_back(512);
	layer.push_back(OUTPUT_SIZE);	
	NetWork nn(layer, NN_INPUT_SIZE, MOM, true);
	nn.outsize = OUTPUT_SIZE;
	assert(nn.read_model(model_name));
	
	//read test data
	
	MatrixXd testX;
	csvToMatrix(testPath, testX, INPUT_SIZE, TEST_READ, NORM);
	matrixExpansion(testX);
	/*
	FILE* out = fopen("out/48_v4.ark","w");
	int idx = 0;
	for(int i=0 ;i<word_size.size();i++){
		fprintf(out,"%s",word_name[i].c_str());
		for(int j =0;j<word_size[i];j++){
			VectorXd tmpv(testX.col(idx));
			fprintf(out,"%d%c",nn.PredictV(tmpv),j==word_size[i]-1?'\n':' ');
			idx++;
		}
	}
	fclose(out);
	*/
	int idxp = 0;
	FILE* fileout = fopen("hmm/probs/test_0.5312_0.4448.out","w");
	for(int i=0 ;i<testX.cols();i+=128){
      int msize = ((i+128<testX.cols())?(i+128):testX.cols())-i;
      assert(msize>0);
      if((i)%10000==0)printf("read: %d\n",i);
			nn.printProbs(testX.block(0,i,testX.rows(),msize),fileout);
	}
	//nn.printProbAll(testX, fileout);
	fclose(fileout);
	
	return 0;

}
