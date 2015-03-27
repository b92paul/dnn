#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<iostream>

using namespace Eigen;
using namespace std;
typedef vector<VectorXd> VXd;

VectorXd sigmoid(VectorXd x)
{
		return (VectorXd((-x).array().exp())+VectorXd::Ones(x.size())).array().inverse();
}
VectorXd sigmoid_prime(VectorXd x)
{
		return (VectorXd::Ones(x.size())-x).cwiseProduct(x);
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
		void back_propgation(VectorXd x,VectorXd y,VectorXd *delta_b,MatrixXd *delta_w)
		{
				vector<VectorXd>activation,zs;
				activation.push_back(x);
				for(int i=0;i<layers;i++) 
				{
						printf("--layers %d\n",i);
						cout << x.size()<<endl;
						cout <<weight[i].size()<<endl;
						cout <<bias[i].size()<<endl;
						x=weight[i]*x+bias[i];
						printf("--layers %d\n",i);
						zs.push_back(x);
						x=sigmoid(x);
						activation.push_back(x);
				}

				VectorXd d= cost_derivative(x,y).cwiseProduct(sigmoid_prime(zs[layers-1]));
				delta_b[layers-1] += d;
				delta_w[layers-1] += d*(x.transpose());
				for(int l=2;l<=layers;l++)
				{
						VectorXd spv = sigmoid_prime(zs[layers-l]);
						d = (weight[layers-l].transpose()*d).cwiseProduct(spv); 
						delta_b[layers-l] += d;
						delta_w[layers-l] += d*(activation[layers+1-l].transpose());
				}
		}
		VectorXd cost_derivative(VectorXd output,VectorXd y){return output-y;}
		void SGD(VXd& TrainX, VXd& TrainY, double eta, int epochs, int msize,VXd& ValX, VXd& ValY ){
			int count =0 ;
			VXd x,y;
			for(int i=0; i<epochs; i++){
				if(count+msize >= TrainX.size())count=0;
				x = VXd(TrainX.begin()+count,TrainX.begin()+count+msize);
				y = VXd(TrainY.begin()+count,TrainY.begin()+count+msize);
				count+=msize;
				printf("eposh %d start\n",i+1);
				update(x,y,eta);
				printf("epoch %d done\n",i+1);
			}
		}
		void update(VXd& bX, VXd& bY,double eta){
				double msize = (double)bX.size();
				VectorXd* delta_b = new VectorXd[layers];
				for(int i=0;i<layers;i++)delta_b[i] = VectorXd::Zero(neuron[i]);
				MatrixXd* delta_w = new MatrixXd[layers];
				
				for(int i=0;i<layers;i++){
					if(i==0) delta_w[i] = MatrixXd::Zero(input_size,neuron[i]);
					else delta_w[i] = MatrixXd::Zero(neuron[i-1],neuron[i]);
				}
				for(int i=0;i<bX.size();i++){
					printf("backprop %d\n",i);
					back_propgation(bX[i],bY[i],delta_b,delta_w);
				}
				for(int i=0;i<layers;i++){
					bias[i] -= eta*delta_b[i]/msize;
					weight[i] -= eta*delta_w[i]/msize;	
				}
		}
		double eval(VXd ValBatchX){return 0.0;}

};
