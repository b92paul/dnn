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
				record = new VectorXd[bptt];
				srand(time(NULL));
				for(int i=0;i<layers;i++) 
				{
						neuron[i]=Neuron[i];
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = RandomMat(neuron[i],num)/ sqrt((double)num);
						if(i<layers-1) 
						{
								recur_weight[i] = RandomMat(neuron[i],neuron[i])/ sqrt((double)neuron[i]);
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
				activation[0]=x;
				for(int i=0;i<layers;i++) {
						if(i<layers-1) zs[i]=weight[i]*activation[i]+recur_weight[i]*activation[i+1] ;
						else zs[i]=weight[i]*activation[i];
						activation[i+1]=sigmoid(zs[i]);
				}
				//cost function
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
						VectorXd tmp =record[index-i-1].array()*(1-record[index-i+1].array()); 
						delta[0] = 
							(recur_weight[0].T()*delta[0]).cwiseProduct(tmp);
				}
		}
		void cost_derivative(const VectorXd& a,const VectorXd& y,VectorXd& output){output=a-y;return;}
		
		void SGD(VectorXd **BX,VectorXd **BY,int data_length,int *sentence_len,double eta,int epochs,int msize,
				VectorXd **ValX=NULL,VectorXd **ValY=NULL,int Val_len=0,int *Val_sen_len=NULL,VectorXd **testX=NULL,bool isPredict=false)
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
				printf("%d count=%d end=%d QQ\n",i,count,end);
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
					//Predict(testX);
				}
			}
			delete [] BatX;
			delete [] BatY;
			delete [] bat_sentence_len;
		}
		void update(VectorXd **BX, VectorXd **BY,int *senten_len,int len,double eta,int time){
			for(int i=0;i<len;i++)
			{
				for(int j=1;j<layers;j++) activation[j] = VectorXd::Zero(neuron[j]);  
				for(int j=0;j<senten_len[i];j++)
				{
					fast_back_propagation(BX[j][i],BY[j][i],delta_w,delta_recur_w,j);
					record[j]= activation[1];
				}
			}
			for(int i=0;i<layers;i++){
				weight[i].noalias() -= (eta*delta_w[i])/len;	
				if(i != layers-1) recur_weight[i].noalias() -= (eta*delta_recur_w[i])/len;
			}
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
		/*
		void Predict(MatrixXd& testX){
			puts("--predict!!");
			VXd testY(testX.cols());	
			for(int i=0;i<testX.cols();i++){
				if((i+1)%20000 ==0) printf("predict test:%d\n",i);
				testY[i] = feedforward(testX.col(i));
			}
			if(outsize != testY[0].size()){puts("error!!");return;}

			char buf[10000],buf2[10000];
			int id;
			
			char lmap_path[] = "../../data/merge/lmap2.out";
			char lmap_path_39[] = "../../data/merge/lmap_392.out";
			char testId[]    = "../../data/merge/test_id2.out";
			char map48_39[] = "../../data/phones/48_39.map";

			// read lmap
			vector<string> lmap(48);
			FILE* f;
			if(outsize == 48) f = fopen(lmap_path, "r");
			else if(outsize == 39) f = fopen(lmap_path_39,"r");
			else {puts("error!!");return;}

			while(~fscanf(f, "%d %s",&id,buf)){
				lmap[id] = string(buf);
			}
			fclose(f);
			puts("done read lmap");

			// read testId	
			vector<string> name;
			f = fopen(testId,"r");
			while(~fscanf(f,"%s",buf)){
				name.push_back(buf);
			}
			fclose(f);
			puts("done read testId");

			// read 48 to 39
			map<string,string> mp; 
			if(outsize == 48){

				f = fopen(map48_39,"r");
				while(~fscanf(f,"%s %s",buf,buf2)){
					mp[string(buf)] = string(buf2);
				}
				fclose(f);
				puts("done read 48_39map");
			
			}

			//filename
			string output_file = string("");
			string output_dir = string("out/");
			//timestamp
			time_t rawtime;
			struct tm * timeinfo;
			time ( &rawtime );
			timeinfo = localtime ( &rawtime );
			//e_val e_in in filename
			sprintf(buf, "%.4f_%.4f_",e_val,e_in);
			output_file += buf;
			sprintf(buf, "%s_t",asctime (timeinfo));
			output_file += string(buf);
			f = fopen((output_dir+output_file+".csv").c_str(),"w");
			//output to file
			fprintf(f,"Id,Prediction\n");
			for(int i=0;i<testY.size();i++){
				if(outsize == 48)
					fprintf(f,"%s,%s\n",name[i].c_str(),mp[lmap[max_number(testY[i])]].c_str());
				else if(outsize == 39)
					fprintf(f,"%s,%s\n",name[i].c_str(),lmap[max_number(testY[i])].c_str());
				else {puts("error!!");fclose(f);return;}
			}
			fclose(f);
			puts("--predict done!!");
		}*/
		
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
				//delta_w_old = new MatrixXd[layers];
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
