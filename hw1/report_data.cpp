#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <algorithm>
#include "fast_dnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label_39.out";
char trainPath[] = "../../data/merge/train.out";
#define INPUT_SIZE 69
#define OUTPUT_SIZE 39 
#define TRAIN_READ 1124823 // Max is 1124823
#define TEST_READ 180406 // Max is 180406
#define MOM 0.0
#define ETA 0.4
#define BATCH_NUM 50000000 // 1 epoch about 1e6 data
#define BATCH_SIZE 500
#define VAL_SIZE 10000
#define TIME_DECAY false
#define TIME_DECAY_NUM -1
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
#define Mat MatrixXd
#define pb push_back
struct Data{
	MatrixXd inputX, inputY;
	MatrixXd valX, valY;
	void read(char *trainPath=::trainPath){
		csvToMatrix(trainPath, inputX, INPUT_SIZE, TRAIN_READ);
		csvToMatrix(labelPath, inputY,OUTPUT_SIZE, TRAIN_READ);
		shuffleMatrix(inputX, inputY, valX, valY, VAL_SIZE);
		printf("X size = %lu\n",inputX.rows());
		printf("Y size = %lu\n",inputY.rows());
		printf("input X data size = %lu\n",inputX.cols());
		printf("input Y data size = %lu\n",inputY.cols());
		printf("val X data size = %lu\n",valX.cols());
		printf("val Y data size = %lu\n",valY.cols());
	}
};
map<int,Data>DataMap;

struct Model {
	vector<int>layer;
	double eta,mom;
	int input_len,batch_size;
	int epochs;
	void read(FILE *f=stdin) {
		int layer_size;
		fscanf(f,"%d",&layer_size);
		while(layer_size--) {
			int a;
			fscanf(f,"%d",&a);
			layer.pb(a);
		}
		fscanf(f,"%lf%lf",&eta,&mom);
		fscanf(f,"%d",&input_len);
		fscanf(f,"%d%d",&batch_size,&epochs);
	}
};
void run(Model &m) {

}
void work(char *filename) {
  FILE *f = fopen(filename,"r");
	if(f == NULL) {
		printf("file %s not found\n",filename);
		exit(1);
	}
	int T;
	fscanf(f,"%d",&T);
	printf("%d test cases\n",T);
	char out[1000];
	sprintf(out,"%s.out",filename);
	FILE *w = fopen(out,"w");
	while(T--){
		Model m;
		m.read(f);
		run(m);
	}
	fclose(f);
	fclose(w);
}
int main(int argc,char **argv){
	if(argc!=2) {
		printf("USAGE: %s input_file_name\n",argv[0]);
		return 0;
	}
	work(argv[1]);
	return 0;
	// new network
	/*vector<int> layer;
	layer.push_back(200);
	layer.push_back(150);
	layer.push_back(150);
	layer.push_back(150);
	layer.push_back(150);
	layer.push_back(OUTPUT_SIZE);	
	NetWork nn(layer, INPUT_SIZE, MOM, true);
	nn.outsize = OUTPUT_SIZE;
	
	//read test data
	MatrixXd testX;
	csvToMatrix(testPath, testX, INPUT_SIZE, TEST_READ);
	printf("test data size = %lu\n",testX.cols());
	nn.SGD(inputX, inputY, ETA, BATCH_NUM, BATCH_SIZE, TIME_DECAY, TIME_DECAY_NUM, valX, valY, testX);

	//nn.Predict(testX);
	*/
	return 0;

}
