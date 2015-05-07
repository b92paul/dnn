#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "new_fast_dnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label_392.out";
char trainPath[] = "../../data/merge/f_train2.out";
char testPath[]  = "../../../data/test_0.ark";
char testId[]    = "../../data/merge/test_id2.out";
#define INPUT_SIZE 69
#define NN_INPUT_SIZE 345
#define OUTPUT_SIZE 48 
#define TRAIN_READ 112482 // Max is 1124823
#define TEST_READ 180406 // Max is 180406
#define MOM 0.0
#define ETA 0.1
#define BATCH_NUM 50000000 // 1 epoch about 1e6 data
#define BATCH_SIZE 128
#define VAL_SIZE 24823
#define TIME_DECAY true
#define TIME_DECAY_NUM 500000.0
#define NORM 1.7
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
void csvToMatrix(char* filename,  MatrixXd& out,int length, int cut){
	int idx = 0;
	out = MatrixXd(length, cut);
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	double num;char split;
	if(csv == NULL) puts("no such file !!!");
	VectorXd tmpXd;
	vector<double> tmp;
	int T;
	int xsize,word_len;
	fscanf(csv,"%d\n",&T);
	while(~fscanf(csv,"%d %d\n",&word_len,&xsize)){
		word_size.push_back(word_len);
		fgets(buf,10000,csv);
		word_name.push_back(string(buf));
		printf("%d,%s",word_len,buf);
		for(int i=0;i<word_len;i++){
			for(int j=0;j<xsize;j++){
				fscanf(csv,"%lf%c",&num,&split);
				tmp.push_back(num);
			}
			tmpXd = VectorXd::Map(&tmp[0],tmp.size());
			tmp.clear();
			out.col(idx++) << tmpXd/NORM;
			if((idx)%check ==0) printf("red data: %d\n",idx);
		}
		fgets(buf,10000,csv);
		printf("%s",buf);
	}
	if(cut > out.cols()) out.conservativeResize(Eigen::NoChange, cut);
	fclose(csv);
}
void matrixExpansion(MatrixXd& A){
	printf("%lu %lu\n",A.rows(),A.cols());
	MatrixXd tmp(A.rows()*5, A.cols());
	int len = A.rows();
	int total = A.cols();
	for(int i=0; i<total;i++){
		for(int j=0 ;j<=4;j++){
			int idx = (i - 4 + j*2+total) % total;
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
	nn.read_model("models/48_v4.out");
	
	//read test data
	
	MatrixXd testX;
	csvToMatrix(testPath, testX, INPUT_SIZE, TEST_READ);
	matrixExpansion(testX);
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
	/*
	FILE* fileout = fopen("../../../data/test_prob.out","w");
	nn.printProbAll(testX, fileout);
	fclose(fileout);
	*/
	return 0;

}
