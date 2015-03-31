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
using namespace Eigen;
using namespace std;
typedef vector<VectorXd> VXd;
#define T() transpose()
//#define NDEBUG 1
//color for print
char color[]="\033[0;32m";
char NC[]="\033[0m";
VectorXd sigmoid(VectorXd x)
{
		return (((-x).array().exp())+1).array().inverse();
}
MatrixXd fast_sigmoid(const MatrixXd& x) 
{

		return (((-x).array().exp())+1).array().inverse();
}

void flogistic(const MatrixXd& z, MatrixXd& a){
	a = (((-z).array().exp())+1).array().inverse();
	return;
}
void logistic(const Eigen::MatrixXd& a, Eigen::MatrixXd& z)
{
	double const* aPtr = a.data();
	double const* aEnd = aPtr + a.rows() * a.cols();
	for(double* zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
	{
		if(*aPtr < -45.0)
			*zPtr = 0.0;
		else if(*aPtr > 45.0)
			*zPtr = 1.0;
		else
			*zPtr = 1.0 / (1.0 + std::exp(-*aPtr));
	}
}

void logistic_prime(const MatrixXd& x, MatrixXd& out)
{
		logistic_prime(x,out);
		out = (out.array()*(out.array()+1)).matrix();
}

void s2p(const MatrixXd& x, MatrixXd& out){
	out = (x.array()*(1-x.array()));
	return;
}
MatrixXd fast_sigmoid_prime(const MatrixXd& x)
{
		MatrixXd sg = fast_sigmoid(x);
		return (sg.array()*(1-sg.array())).matrix();
}
class NetWork
{
	public:
		int layers;
		double momentum;
		int *neuron;
		int input_size; // input x size
		VectorXd *bias;
		MatrixXd *weight;
		double e_in, e_val;
		VectorXd* delta_b, *delta_b_old;
		MatrixXd* delta_w, *delta_w_old;
		MatrixXd *zs, *activation;
		MatrixXd* delta;
		bool printTest;
		int outsize = 48;
		NetWork(vector<int>Neuron, int _input_size,double _momentum=0,bool _printTest=false)
						:input_size(_input_size),printTest(_printTest),momentum(_momentum)
		{
				layers = Neuron.size();
				neuron = new int[layers];
				bias = new VectorXd[layers];
				weight = new MatrixXd[layers];
				delta_b = new VectorXd[layers];
				delta_w = new MatrixXd[layers];
				activation = new MatrixXd[layers+1];
				zs = new MatrixXd[layers];
				delta_b_old = new VectorXd[layers];
				delta_w_old = new MatrixXd[layers];
				delta = new MatrixXd[layers];
				srand(time(NULL));
				for(int i=0;i<layers;i++) 
				{
						neuron[i]=Neuron[i];
						bias[i] = VectorXd::Random(neuron[i]) * 3;
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Random(neuron[i],num)/ sqrt((double)num) *3; //sigma -1 ~ 1
						delta_b_old[i] = VectorXd::Zero(neuron[i]);
						if(i==0) delta_w_old[i] = MatrixXd::Zero(neuron[i],input_size);
						else delta_w_old[i] = MatrixXd::Zero(neuron[i],neuron[i-1]);
				}
		}
		~NetWork(){
			delete[] neuron;
			delete[] bias;
			delete[] weight;
			delete[] delta_b;
			delete[] delta_w;
			delete[] zs;
			delete[] activation;
			delete[] delta_b_old;
			delete[] delta_w_old;
			delete[] delta;
		}
		VectorXd feedforward(VectorXd x)
		{
				for(int i=0;i<layers;i++) x = sigmoid(weight[i]*x+bias[i]);
				return x;
		}
		MatrixXd fast_feedforward(MatrixXd x)
		{
				for(int i=0;i<layers;i++) 	flogistic(weight[i]*x+bias[i]*(VectorXd::Ones(x.cols()).T()),x);
				return x;
		}
		void fast_back_propagation(const MatrixXd& x,const MatrixXd& y,VectorXd *delta_b,MatrixXd *delta_w)
		{
				activation[0]=x;
				for(int i=0;i<layers;i++) {
						zs[i]=weight[i]*activation[i]+bias[i]* (VectorXd::Ones(x.cols()).T()) ;
						logistic(zs[i], activation[i+1]);
				}
				
				//cost function
				MatrixXd &d = delta[layers-1];
				cost_derivative(activation[layers],y,d);//.cwiseProduct(fast_sigmoid_prime(zs[layers-1]));
				
				delta_b[layers-1] = d.rowwise().sum();
				delta_w[layers-1] = (d* activation[layers-1].T());
				for(int l=2;l<=layers;l++) {
						MatrixXd& d = delta[layers-l];
						s2p(activation[layers-l+1], d);
						d = (weight[layers-l+1].T()*delta[layers-l+1]).cwiseProduct(d); 
						delta_b[layers-l] = d.rowwise().sum();
						delta_w[layers-l] = (d * activation[layers-l].T());
				}
		}

		MatrixXd fast_cost_derivative(MatrixXd& output,const MatrixXd& y){return output-y;}
		void cost_derivative(const MatrixXd& a,const MatrixXd& y,MatrixXd& output){output=a-y;return;}

		void SGD(MatrixXd& BX, MatrixXd& BY, double eta, int epochs, int msize,VXd& ValX, VXd& ValY ,VXd& testX, 
										bool findModel = false, vector<int>* param = NULL,
										vector<pair<double,double> >* ans = NULL){
			int count =0 ,end=msize;
			puts("-- Start SGD.");
			for(int l=1;l<= layers;l++){
				activation[l] = MatrixXd(neuron[l-1],msize);
				delta[l-1] = MatrixXd(neuron[l-1],msize);
			}
			clock_t start_time = clock();
			for(int i=0; i<epochs; i++){
				if(end > BX.cols())count=0,end=msize;
				
				int num = 1000;
				if(findModel){
					num = (*param)[0];
					if(i == (*param)[1])break;
				}
				//update by back propagation
				update(BX.block(0,count,BX.rows(),msize),BY.block(0,count,BY.rows(),msize),eta,i);
				
				if((i+1)%num == 0){
					e_val = eval(ValX,ValY);
					e_in = fast_eval(BX.block(0,count,BX.rows() ,msize),BY.block(0,count,BY.rows(),msize));
					// print exp message
					printf("%s-- Spend %f time to train %d batch.\n",
													color,((float)(clock()-start_time))/CLOCKS_PER_SEC,num);
					printf("Layer number = %d; ",layers);
					for(int i=0;i<layers;i++)printf("%d%c",neuron[i],i==(layers-1)?'\n':',');
					printf("learning rate = %.3f, momentum = %.3f\n",eta,momentum);
					printf("e_val = %lf\n",e_val);
					printf("e_in of batch = %lf\n",e_in);
					printf("-- batch %d done.%s\n",i+1,NC);
					//return parameter for model finder
					if(findModel)ans->push_back(make_pair(e_val,e_in));
					start_time = clock();	
				}
				count+=msize, end+=msize;
				if(printTest && (i+1)%5000 == 0){
					char model_name[]="model.QAQ";
					save_model(model_name);
					Predict(testX);
				}
			}
		}
		void update(Block<MatrixXd> BX, Block<MatrixXd> BY,double eta,int time){
				double msize = (double)BX.cols();
				if(e_val>=0.55) eta/=8;
				else if(e_val>=0.53) eta/=4;
				else if(e_val>=0.5) eta/=2;
				if(time%2==0)
				{
					fast_back_propagation(BX,BY,delta_b,delta_w);
					for(int i=0;i<layers;i++){
						bias[i].noalias() -= (eta*delta_b[i]+delta_b_old[i]*momentum)/msize;
						weight[i].noalias() -= (eta*delta_w[i]+delta_w_old[i]*momentum)/msize;	
					}
				}
				else {
					fast_back_propagation(BX,BY,delta_b_old,delta_w_old);
					for(int i=0;i<layers;i++){
						bias[i].noalias() -= (eta*delta_b_old[i]+delta_b[i]*momentum)/msize;
						weight[i].noalias() -= (eta*delta_w_old[i]+delta_w[i]*momentum)/msize;	
					}
				}
		}
		double eval(VXd& ValBatchX,VXd& ValBatchY)
		{
				int num=0;
				int binN[49]={}, binY[49]={};
				for(int i=0;i<ValBatchX.size();i++)
				{
						VectorXd output = feedforward(ValBatchX[i]);
						int now = max_number(output);
						binN[now]++;
						if(now == max_number(ValBatchY[i])) num++,binY[now]++;
				}
				for(int i=0;i<49;i++){
					printf("Idx=%2d:%4d in %5d| ",i, binY[i], binN[i]);
					if((i+1) % 7 == 0)puts("");
				}
				return (double)num/ValBatchX.size();
		}
		double fast_eval(Block<MatrixXd> ValBatchX,Block<MatrixXd> ValBatchY)
		{
				int num=0;
				for(int i=0;i<ValBatchX.cols();i++)
				{
						VectorXd output = feedforward(ValBatchX.col(i));
						if(max_number(output) == max_number(ValBatchY.col(i))) num++;
				}
				return (double)num/ValBatchX.cols();
		}
				
		int max_number(const VectorXd& y)
		{
				VectorXd::Index maxIndex;
				y.maxCoeff(&maxIndex);
				return int(maxIndex);
		}
		void Predict(VXd& testX){
			puts("--predict!!");
			VXd testY(testX.size());	
			for(int i=0;i<testX.size();i++){
				if((i+1)%20000 ==0) printf("predict test:%d\n",i);
				testY[i] = feedforward(testX[i]);
			}
			if(outsize != testY[0].size()){puts("error!!");return;}

			char buf[10000],buf2[10000];
			int id;
			
			char lmap_path[] = "../../data/merge/lmap.out";
			char lmap_path_39[] = "../../data/merge/lmap_39.out";
			char testId[]    = "../../data/merge/test_id.out";
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
				for(int i=0;i<layers;i++) file<<bias[i]<<endl<<weight[i]<<endl;
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
				delete [] neuron,bias,weight,delta_b,delta_w,delta_b_old,delta_w_old,zs,activation;
				neuron = new int[layers];
				bias = new VectorXd[layers];
				weight = new MatrixXd[layers];
				delta_b = new VectorXd[layers];
				delta_w = new MatrixXd[layers];
				activation = new MatrixXd[layers+1];
				zs = new MatrixXd[layers];
				delta_b_old = new VectorXd[layers];
				delta_w_old = new MatrixXd[layers];
				for(int i=0;i<layers;i++) file>>neuron[i];
				for(int i=0;i<layers;i++) 
				{
						delta_b_old[i] = VectorXd::Zero(neuron[i]);
						if(i==0) delta_w_old[i] = MatrixXd::Zero(neuron[i],input_size);
						else delta_w_old[i] = MatrixXd::Zero(neuron[i],neuron[i-1]);
				}
				for(int i=0;i<layers;i++) 
				{
						bias[i] = VectorXd::Zero(neuron[i]);
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Zero(neuron[i],num);
				}
				for(int i=0;i<layers;i++)
				{	
						for(int j=0;j<neuron[i];j++) file>>bias[i](j);
						int num=input_size;
						if(i!=0) num=neuron[i-1];
						for(int k=0;k<neuron[i];k++)
							for(int j=0;j<num;j++)
									file>>weight[i](k,j);
				}
				file.close();
				return true;
		}
};
