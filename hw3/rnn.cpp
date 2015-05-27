#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<map>
#include<time.h>
#include<ctime>
#include<sys/time.h>
#include<random>
using namespace Eigen;
using namespace std;
typedef vector<VectorXd> VXd;
#define T() transpose()
//#define NDEBUG 1
mt19937 rng(0x514514);
int randint(int lb, int ub) {
	return uniform_int_distribution<int>(lb, ub)(rng);
}
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);
MatrixXd RandomMat(int row, int col){
	MatrixXd res(row,col);
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++) res(i,j) = distribution(generator);
	}
	return res;
}
VectorXd RandomVet(int row){
	VectorXd res(row);
	for(int i=0;i<row;i++){
		res(i) = distribution(generator);
	}
	return res;
}
VectorXd sigmoid(VectorXd x)
{
		return (((-x).array().exp())+1).array().inverse();
}

void shuffleTrain(MatrixXd& X,MatrixXd& Y){
	puts("-- In shuffle Train");
	int n = X.cols();
	for(int i = n-1; i >= 0; i--){
		int idx = randint(0,i);
		X.col(i).swap(X.col(idx));
		Y.col(i).swap(Y.col(idx));
	}
	puts("-- Done shuffle Tatrix");
}
/*void s2p(const VectorXd& x, VectorXd& out){
	out = (x.array()*(1-x.array()));
	return;
}*/

class NetWork
{
	public:
		int layers;
		double momentum;
		int *neuron;
		int input_size; // input x size
		MatrixXd *weight;
		MatrixXd *recur_weight;
		double e_in, e_val;
		MatrixXd* delta_w, *delta_w_old,*delta_recur_w,*delta_recur_w_old;
		VectorXd *zs, *activation;
		VectorXd* delta;
		bool printTest;
		//int outsize = 48;
		int bptt=8;
		VectorXd *record;
		NetWork(vector<int>Neuron, int _input_size,double _momentum=0,bool _printTest=false)
						:input_size(_input_size),printTest(_printTest),momentum(_momentum)
		{
				layers = Neuron.size();
				neuron = new int[layers];
				weight = new MatrixXd[layers];
				recur_weight = new MatrixXd[layers-1];
				delta_w = new MatrixXd[layers];
				delta_recur_w = new MatrixXd[layers-1];
				activation = new VectorXd[layers+1];
				zs = new VectorXd[layers];
				delta = new VectorXd[layers];
				record= new VectorXd[100];
				srand(time(NULL));
				for(int i=0;i<layers;i++) 
				{
						neuron[i]=Neuron[i];
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = RandomMat(neuron[i],num)/ sqrt((double)num);
						delta_w[i] = MatrixXd::Zero(neuron[i],num);
						if(i<layers-1) 
						{
								recur_weight[i] = RandomMat(neuron[i],neuron[i])/ sqrt((double)neuron[i]);
							  delta_recur_w[i] = MatrixXd::Zero(neuron[i],neuron[i]);
						}
				}
				puts("initial done");
		}
		~NetWork(){
			delete[] neuron;
			delete[] weight;
			delete[] recur_weight;
			delete[] delta_recur_w; 
			//delete[] delta_recur_w_old; 
			delete[] delta_w;
			delete[] zs;
			delete[] activation;
			//delete[] delta_w_old;
			delete[] delta;
			delete [] record;
		}
		VectorXd feedforward(VectorXd x)
		{
				for(int i=0;i<layers;i++) 
				{
					if(i<layers-1) x = sigmoid(weight[i]*x+recur_weight[i]*activation[i+1]);
					else x = sigmoid(weight[i]*x);
				}
				return x;
		}
		void fast_back_propagation(const VectorXd& x,const VectorXd& y,MatrixXd *delta_w,MatrixXd *delta_recur_w,int index)
		{
				puts("QAQAQ 1");
				activation[0]=x;
				puts("QAQAQ 1.5");
				for(int i=0;i<layers;i++) {
						if(i<layers-1) zs[i]=weight[i]*activation[i]+recur_weight[i]*activation[i+1] ;
						else zs[i]=weight[i]*activation[i];
						activation[i+1]=sigmoid(zs[i]);
				}
				VectorXd &d = delta[layers-1];
				cost_derivative(activation[layers],y,d);//.cwiseProduct(fast_sigmoid_prime(zs[layers-1]));
				delta_w[layers-1] += (d* activation[layers-1].T());
				for(int l=2;l<=layers;l++) {
						VectorXd& d = delta[layers-l];
						d = (activation[layers-l+1].array()*(1-activation[layers-l+1].array()));
						d = (weight[layers-l+1].T()*delta[layers-l+1]).cwiseProduct(d); 
						delta_w[layers-l] += (d * activation[layers-l].T());
				}
				for(int i=0;i<min(index,bptt);i++)
				{
						delta_recur_w[0] += (delta[0]* record[index-i-1].T());
						VectorXd tmp =record[index-i-1].array()*(1-record[index-i-1].array()); 
						delta[0] = 
							(recur_weight[0].T()*delta[0]).cwiseProduct(tmp);
				}
		}
		void cost_derivative(const VectorXd& a,const VectorXd& y,VectorXd& output){output=a-y;return;}
		
		void SGD(VectorXd **BX,VectorXd **BY,int data_length,int *sentence_len,double eta,int epochs,int msize,
				VectorXd **ValX=NULL,VectorXd **ValY=NULL,int Val_len=0,int *Val_sen_len=NULL,VectorXd **testX=NULL,VectorXd **testY=NULL,int test_len=0,int *test_sen_len=NULL,bool isPredict=false)
		{
			int count=0,end=msize;
			puts("--Start SGD.--");
			clock_t start_time = clock();
			struct timeval tstart, tend;
			gettimeofday(&tstart, NULL);
			VectorXd **BatX, **BatY;
			int *bat_sentence_len;
			BatX = new VectorXd*[msize];
			BatY = new VectorXd*[msize];
			bat_sentence_len = new int[msize];
			printf("batch = %d\n data_len=%d\n",epochs,data_length);
			for(int i=0;i<epochs;i++)
			{
				if(end >= data_length)
				{
						count = 0;
						end = msize;
				}
				printf("%d QQ\n",i);
				for(int j=count;j<end;j++)
				{
					BatX[j-count] = BX[j];
					BatY[j-count] = BY[j];
					bat_sentence_len[j-count] = sentence_len[j];
				}
				printf("%d QQ\n",i);
				update(BatX,BatY,bat_sentence_len,end-count,eta,i);
				printf("%d QQ\n",i);
				int num = 50000;
				if((i+1)%num == 0)
				{
					e_val = eval_vec(ValX,ValY,Val_sen_len,Val_len);
					e_in=eval_vec(BatX,BatY,bat_sentence_len,end-count);
					gettimeofday(&tend, NULL);
					double time_delta = ((tend.tv_sec  - tstart.tv_sec) * 1000000u + tend.tv_usec - tstart.tv_usec) / 1.e6;
					printf("Spend %f time to train %d batch.\n",(time_delta),num);
					printf("Layer number = %d; ",layers);
					for(int i=0;i<layers;i++)printf("%d%c",neuron[i],i==(layers-1)?'\n':',');
					printf("learning rate = %.3f, momentum = %.3f\n",eta, momentum);
					printf("e_val = %lf\n",e_val);
					printf("e_in of batch = %lf\n",e_in);
					printf("-- batch %d done.\n",i+1);
					start_time = clock();	
					gettimeofday(&tstart, NULL);
				}
				printf("%d QQ\n",i);
				count+=msize, end+=msize;
				if(isPredict && (i+1)%50000 == 0){
					char model_name[]="model.QAQ";
					save_model(model_name);
					Predict(testX,testY,test_len,test_sen_len);
				}
			}
			delete [] BatX;
			delete [] BatY;
			delete [] bat_sentence_len;
		}
		void update(VectorXd **BX, VectorXd **BY,int *senten_len,int len,double eta,int time){
			for(int i=0;i<len;i++)
			{
				activation[0] = VectorXd::Zero(200);
				for(int j=0;j<layers;j++)
				{
					if(j!=layers-1)activation[j+1] = VectorXd::Zero(neuron[j]);  
					zs[j] = VectorXd::Zero(neuron[j]);
				}
				for(int j=0;j<senten_len[i];j++)
				{
					printf("OAO OAO 1\n");
					if(BX[i][j].size() != 200) printf("--------- %d\n",BX[i][j].size());
					fast_back_propagation(BX[i][j],BY[i][j],delta_w,delta_recur_w,j);
					printf("OAO OAO 2\n");
					record[j]= activation[1];
					printf("OAO OAO 3\n");
				}
			}
			printf("update done\n");
			for(int i=0;i<layers;i++){
				weight[i].noalias() -= (eta*delta_w[i])/len;	
				if(i != layers-1) recur_weight[i].noalias() -= (eta*delta_recur_w[i])/len;
			}
			printf("calc done\n");
		}
		
		double eval_vec(VectorXd **ValBatchX,VectorXd **ValBatchY,int *senten_len,int len)
		{
				double sum = 0;
				for(int i=0;i<len;i++)
				{
						for(int j=0;j<senten_len[i];j++)
						{
								VectorXd out=feedforward(ValBatchX[i][j]);
								out.normalize();
								ValBatchY[i][j].normalize();
								double dot = out.dot(ValBatchY[i][j]);
								sum += dot;
						}		
				}
				return sum;
		}
		double eval_1ofN(VectorXd **ValBatchX,VectorXd **ValBatchY,int *senten_len,int len)
		{
				int num=0,total=0;
				for(int i=0;i<len;i++)
				{
						for(int j=0;j<senten_len[i];j++)
						{
								VectorXd out=feedforward(ValBatchX[i][j]);
								if(max_number(out) == max_number(ValBatchY[i][j])) num++;
								total++;
						}		
				}
				return (double)num/(double)total;
		}
		
		int max_number(const VectorXd& y)
		{
				VectorXd::Index maxIndex;
				y.maxCoeff(&maxIndex);
				return int(maxIndex);
		}
		
		bool save_model(char file_name[])
		{
				char total_file_name[100] ="saved_models/";
				fstream file;
				strcat(total_file_name,file_name);
				file.open(total_file_name,ios::out);
				if(file.fail()) return false;
				file<<input_size<<endl;
				file<<layers<<endl;
				for(int i=0;i<layers;i++) 
				{
					file<<neuron[i];
					if(i!=layers-1) file<<" ";
					else file<<endl;
				}
				for(int i=0;i<layers;i++) file<<weight[i]<<endl;
				for(int i=0;i<layers-1;i++) file<<recur_weight[i]<<endl;
				file.close();
				return true;
		}
		double my_norm(double a){	return (a+1)/2;	}
	  void Predict(VectorXd **testX,VectorXd **testY,int length,int *test_senten_len)
		{
				int *ans;
				ans = new int[5];
				FILE *out = fopen("output.csv","w");
				fprintf(out,"id,answer\n");
				for(int i=0;i<length;i+=5)
				{
						for(int j=0;j<5;j++) ans[j] = 0;
						for(int j=i;j<i+5;j++)
						{
								for(int k=0;k<test_senten_len[j];k++)
								{
										VectorXd out=feedforward(testX[j][k]);
										out.normalize();
										testY[i][j].normalize();
										double norm = my_norm(out.dot(testY[j][k]));
										ans[j-i] +=norm;
								}
						}
						double max=-1;
						int index = 0;
						for(int j=0;j<5;j++)
						{
								if(max<ans[j])
								{
										max=ans[j];
										index=j;
								}
						}
						fprintf(out,"%d,%c\n",i+1,'a'+index);
				}
		}
		bool read_model(char file_name[])
		{
				char total_file_name[100] ="saved_models/";
				fstream file;
				strcat(total_file_name,file_name);
				file.open(total_file_name,ios::in);
				if(file.fail()) return false;
				file>>input_size;
				file>>layers;
				delete [] neuron;
				delete [] weight;
				delete [] recur_weight;
				delete [] delta_w;
				delete [] delta_recur_w;
				delete [] zs;
				delete [] activation;
				neuron = new int[layers];
				weight = new MatrixXd[layers];
				delta_w = new MatrixXd[layers];
				recur_weight = new MatrixXd[layers-1];
				delta_recur_w = new MatrixXd[layers-1];
				activation = new VectorXd[layers+1];
				zs = new VectorXd[layers];
				for(int i=0;i<layers;i++) file>>neuron[i];
				for(int i=0;i<layers;i++) 
				{
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Zero(neuron[i],num);
						if(i!=layers-1) recur_weight[i] = MatrixXd::Zero(neuron[i],neuron[i]);
				}
				for(int i=0;i<layers;i++)
				{	
						int num=input_size;
						if(i!=0) num=neuron[i-1];
						for(int k=0;k<neuron[i];k++)
							for(int j=0;j<num;j++)
									file>>weight[i](k,j);
				}
				for(int i=0;i<layers-1;i++)
				{	
						for(int k=0;k<neuron[i];k++)
							for(int j=0;j<neuron[i];j++)
									file>>weight[i](k,j);
				}
				file.close();
				return true;
		}
};
