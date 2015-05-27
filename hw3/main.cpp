#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include "rnn.cpp"
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
#define INPUT_SIZE 200
#define OUTPUT_SIZE 200 

typedef vector<double> VD;
typedef vector<VectorXd> VXd;
int check = 100000;

void read_data(char *filename,VectorXd **X,VectorXd **Y,int *data_length,int *sentence_len,int length=0)
{
	vector<double> v;
	FILE *in = fopen(filename,"r");
	int tmp,len;
	double value;
	fscanf(in,"%d",&len);
	if(length !=0 )len = length;
	*data_length = len;
	for(int i=0;i<len;i++)
	{
			fscanf(in,"%d",&tmp);
			if(i % 5000 == 0)printf("now %d\n",i); 
			sentence_len[i] = tmp-1;
			X[i] = new VectorXd[tmp-1];
			Y[i] = new VectorXd[tmp-1];
			for(int j=0;j<tmp;j++)
			{
					for(int k=0;k<200;k++)
					{
							fscanf(in,"%lf",&value);
							v.push_back(value);
					}
					if(v.size() < 200 ) printf("%d\n",v.size());
				  if(j!=tmp-1) X[i][j] = VectorXd::Map(&v[0],v.size());
				  if(j!=0) Y[i][j-1] = VectorXd::Map(&v[0],v.size());
					v.clear();
			}
	}
	puts("read training data done!");
	fclose(in);
}

int main(int argc,char **argv){
	VectorXd **inputX;
	VectorXd **inputY;
	VectorXd **valX;
	VectorXd **valY;
	VectorXd **testX;
	VectorXd **testY;
	int data_length=0,val_data_length=0,test_data_length=0;
	int *sentence_len,*val_senten_len,*test_senten_len;
	double eta=0.5;
	int msize=10;
	if(argc <= 3) 
	{
			printf("need training data path ,validation data path ,sentece length,testing data path\n");
			return 0 ;
	}
	int len = 0;
	len = atoi(argv[3]);
	inputX = new VectorXd*[len];//幾句
	inputY = new VectorXd*[len];
	sentence_len = new int[len];
	valX = new VectorXd*[3000];//幾句
	valY = new VectorXd*[3000];
	val_senten_len = new int[3000];
	int epochs= len/10;
	read_data(argv[1],inputX,inputY,&data_length,sentence_len,len);
	read_data(argv[2],valX,valY,&val_data_length,val_senten_len,3000);
	read_data(argv[4],testX,testY,&test_data_length,test_senten_len);
	// new network
	vector<int> layer;
	layer.push_back(150);
	layer.push_back(OUTPUT_SIZE);	
	NetWork nn(layer, INPUT_SIZE);
	nn.SGD(inputX,inputY,data_length,sentence_len,eta,epochs,msize,valX,valY,val_data_length,val_senten_len,testX,testY,test_data_length,test_senten_len);
	for(int i=0;i<data_length;i++)
	{
			delete [] inputX[i];
			delete [] inputY[i];
	}
	for(int i=0;i<val_data_length;i++)
	{
			delete [] valX[i];
			delete [] valY[i];
	}
	for(int i=0;i<test_data_length;i++)
	{
			delete [] testX[i];
			delete [] testY[i];
	}
	delete [] inputX;
	delete [] inputY;
	delete [] valX;
	delete [] valY;
	delete [] testX;
	delete [] testY;
	delete [] sentence_len;
	delete [] val_senten_len;
	delete [] test_senten_len;
	return 0;
}
