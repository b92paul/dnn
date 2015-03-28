#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<iostream>

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
						bias[i] = VectorXd::Random(neuron[i]);
						int num;
						if(i==0) num=input_size;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Random(neuron[i],num); //-1 ~ 1
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
		void fast_back_propgation(MatrixXd x,MatrixXd y,VectorXd *delta_b,MatrixXd *delta_w)
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
				MatrixXd d= fast_cost_derivative(x,y).cwiseProduct(fast_sigmoid_prime(zs[layers-1]));
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
		void SGD(VXd& TrainX, VXd& TrainY, double eta, int epochs, int msize,VXd& ValX, VXd& ValY ){
			int count =0 ,end=msize;
			VXd x,y;
			VXd judge;
			for(int i=0; i<epochs; i++){
				printf("-- epoch %d start\n",i);
				if(end >= TrainX.size())count=0,end=msize;
				
				MatrixXd BX(TrainX[0].size(),msize);
				MatrixXd BY(TrainY[0].size(),msize);
				
				for(int i=count;i< end; i++){
					BX.col(i-count) << TrainX[i];
					BY.col(i-count) << TrainY[i];	
				}
				
				//x = VXd(TrainX.begin()+count,TrainX.begin()+count+msize);
				//y = VXd(TrainY.begin()+count,TrainY.begin()+count+msize);
				count+=msize;
				end+=msize;
				update(BX,BY,eta);
				printf("e_val = %lf\n",eval(ValX,ValY));
				printf("e_in of batch = %lf\n",fast_eval(BX,BY));
				printf("-- epoch %d done \n",i);

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
				
				fast_back_propgation(BX,BY,delta_b,delta_w);
				
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
						if(y(i)>max)
						{
								max=y(i);
								num=i;
						}
				}
				return num;
		}
};
