#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#include<string>
#include<map>
#include<time.h>
using namespace Eigen;
using namespace std;
typedef vector<VectorXd> VXd;
#define T() transpose()
VectorXd sigmoid(VectorXd x)
{
		return (VectorXd((-x).array().exp())+VectorXd::Ones(x.size())).array().inverse();
}
MatrixXd fast_sigmoid(MatrixXd x)
{
		return (MatrixXd((-x).array().exp())+MatrixXd::Ones(x.rows(),x.cols())).array().inverse();
}

VectorXd sigmoid_prime(VectorXd x)
{
		VectorXd sg = sigmoid(x);
		return (VectorXd::Ones(x.size())-sg).cwiseProduct(sg);
}
MatrixXd fast_sigmoid_prime(MatrixXd x)
{
		MatrixXd sg = fast_sigmoid(x);
		return (MatrixXd::Ones(x.rows(),x.cols())-sg).cwiseProduct(sg);
}
class NetWork
{
	public:
		int layers;
		int *neuron;
		int input_size; // input x size
		VectorXd *bias;
		MatrixXd *weight;
		
		NetWork(vector<int>Neuron, int _input_size):input_size(_input_size)
		{
				layers = Neuron.size();
				neuron = new int[layers];
				bias = new VectorXd[layers];
				weight = new MatrixXd[layers];
				srand(time(NULL));
				for(int i=0;i<layers;i++) 
				{
						neuron[i]=Neuron[i];
						bias[i] = VectorXd::Random(neuron[i]) * 3;
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Random(neuron[i],num)/ sqrt((double)num) *3; //sigma -1 ~ 1
				}
		}
		VectorXd feedforward(VectorXd x)
		{
				for(int i=0;i<layers;i++) 	x=sigmoid(weight[i]*x+bias[i]);
				return x;
		}
		MatrixXd fast_feedforward(MatrixXd x)
		{
				for(int i=0;i<layers;i++) 	x=fast_sigmoid(weight[i]*x+bias[i]*(VectorXd::Ones(x.cols()).T()));
				return x;
		}
		void fast_back_propagation(MatrixXd x,MatrixXd y,VectorXd *delta_b,MatrixXd *delta_w)
		{
				vector<MatrixXd>activation,zs;
				activation.push_back(x);
				for(int i=0;i<layers;i++) 
				{
						x=weight[i]*x+bias[i]* (VectorXd::Ones(x.cols()).T()) ;
						zs.push_back(x);
						x=fast_sigmoid(x);
						activation.push_back(x);
				}
				MatrixXd d= fast_cost_derivative(x,y);//.cwiseProduct(fast_sigmoid_prime(zs[layers-1]));
				delta_b[layers-1] += d.rowwise().sum();
				for(int i=0;i<d.cols();i++){
					delta_w[layers-1] += (d.col(i) * activation[layers-1].col(i).T());
				}
				for(int l=2;l<=layers;l++)
				{
						MatrixXd spv = fast_sigmoid_prime(zs[layers-l]);
						d = (weight[layers-l+1].T()*d).cwiseProduct(spv); 
						delta_b[layers-l] += d.rowwise().sum();
						for(int i=0;i<d.cols();i++){
							delta_w[layers-l] += (d.col(i) * activation[layers-l].col(i).T());
						}
						//delta_w[layers-l] += d*(activation[layers-l].rowwise().sum().T());
				}
		}
		VectorXd cost_derivative(VectorXd output,VectorXd y){return output-y;}
		MatrixXd fast_cost_derivative(MatrixXd output,MatrixXd y){return output-y;}
		void SGD(VXd& TrainX, VXd& TrainY, double eta, int epochs, int msize,VXd& ValX, VXd& ValY ,VXd& testX, 
										bool findModel = false, vector<int>* param = NULL,
										vector<pair<double,double> >* ans = NULL){
			int count =0 ,end=msize;
			VXd x,y;
			VXd judge;
			for(int i=0; i<epochs; i++){
				if(end >= TrainX.size())count=0,end=msize;
				//copy matrix
				MatrixXd BX(TrainX[0].size(),msize);
				MatrixXd BY(TrainY[0].size(),msize);	
				for(int i=count;i< end; i++){
					BX.col(i-count) << TrainX[i];
					BY.col(i-count) << TrainY[i];	
				}
				count+=msize, end+=msize;
				
				int num = 100;
				if(findModel){
					num = (*param)[0];
					if(i == (*param)[1])break;
				}
				
				//update by back propagation
				update(BX,BY,eta);
				
				if((i+1)%num == 0){
					double e_in, e_val;
					e_val = eval(ValX,ValY);
					e_in = fast_eval(BX,BY);
					printf("e_val = %lf\n",e_val);
					printf("e_in of batch = %lf\n",e_in);
					printf("-- batch %d done \n",i);
					if(findModel)ans->push_back(make_pair(e_val,e_in));
				}
				if((i+1)%5000 == 0){
					Predict(testX);
				}
			}
		}
		void update(MatrixXd& BX, MatrixXd& BY,double eta){
				double msize = (double)BX.cols();
				
				VectorXd* delta_b = new VectorXd[layers];
				for(int i=0;i<layers;i++)delta_b[i] = VectorXd::Zero(neuron[i]);
				MatrixXd* delta_w = new MatrixXd[layers];
				
				for(int i=0;i<layers;i++){
					if(i==0) delta_w[i] = MatrixXd::Zero(neuron[i],input_size);
					else delta_w[i] = MatrixXd::Zero(neuron[i],neuron[i-1]);
				}
				
				fast_back_propagation(BX,BY,delta_b,delta_w);
				
				for(int i=0;i<layers;i++){
					bias[i] -= eta*delta_b[i]/msize;
					weight[i] -= eta*delta_w[i]/msize;	
				}
		}
		double eval(VXd ValBatchX,VXd ValBatchY)
		{
				int num=0;
				int binN[49]={};
				int binY[49]={};
				for(int i=0;i<ValBatchX.size();i++)
				{
						VectorXd output = feedforward(ValBatchX[i]);
						int now = max_number(output);
						binN[now]++;
						if(now == max_number(ValBatchY[i])) num++,binY[now]++;
				}
				for(int i=0;i<49;i++){
					printf("idx %3d: %4d in %4d| ",i, binY[i], binN[i]);
					if((i+1) % 7 == 0)puts("");
				}
				return (double)num/ValBatchX.size();
		}
		double fast_eval(MatrixXd ValBatchX,MatrixXd ValBatchY)
		{
				int num=0;
				for(int i=0;i<ValBatchX.cols();i++)
				{
						VectorXd output = feedforward(ValBatchX.col(i));
						if(max_number(output) == max_number(ValBatchY.col(i))) num++;
				}
				return (double)num/ValBatchX.cols();
		}
				
		int max_number(VectorXd y)
		{
				double max=-2147483647;
				int num=-1;
				for(int i=0;i<y.size();i++)
				{
						if(y(i)>= max)
						{
								max=y(i);
								num=i;
						}
				}
				return num;
		}
		void Predict(VXd& testX){
			puts("--predict!!");
			VXd testY(testX.size());	
			for(int i=0;i<testX.size();i++){
				if((i+1)%20000 ==0) printf("predict test:%d\n",i);
				testY[i] = feedforward(testX[i]);
			}

			char buf[10000],buf2[10000];
			int id;
			
			char lmap_path[] = "../../data/merge/lmap.out";
			char testId[]    = "../../data/merge/test_id.out";
			char map48_39[] = "../../data/phones/48_39.map";
			// read lmap
			vector<string> lmap(48);
			FILE* f = fopen(lmap_path, "r");
			while(~fscanf(f, "%d %s",&id,buf)){
				lmap[id] = string(buf);
			}
			puts("done read lmap");

			// read testId	
			vector<string> name;
			f = fopen(testId,"r");
			while(~fscanf(f,"%s",buf)){
				name.push_back(buf);
			}
			puts("done read testId");

			// read 48 to 39
			map<string,string> mp; 
			f = fopen(map48_39,"r");
			while(~fscanf(f,"%s",buf)){
				fscanf(f,"%s",buf2);
				mp[string(buf)] = string(buf2);
			}
			puts("done read 48_39map");
			
			string output_path = string("out/test_label_");
			time_t rawtime;
			struct tm * timeinfo;
			time ( &rawtime );
			timeinfo = localtime ( &rawtime );
			sprintf(buf, "%s_",asctime (timeinfo));

			output_path += string(buf);

			f = fopen((output_path+".csv").c_str(),"w");
			fprintf(f,"Id,Prediction\n");
			for(int i=0;i<testY.size();i++){
				fprintf(f,"%s,%s\n",name[i].c_str(),mp[lmap[max_number(testY[i])]].c_str());
			}
			puts("--predict done!!");
		}
};
