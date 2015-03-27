#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>

using namespace Eigen;
using namespace std;


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
		int input_length;//input x vector_length 
		int layers;
		int *neuron;
		VectorXd *bias;
		MatrixXd *weight;
		
		NetWork(vector<int>Neuron)
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
						if(i==0) num=input_length;
						else num=neuron[i-1];
						weight[i] = MatrixXd::Random(num,neuron[i]); //-1 ~ 1
				}
		}
		VectorXd feedforward(VectorXd x)
		{
				for(int i=0;i<layers;i++) 	x=sigmoid(weight[i]*x+bias[i]);
				return x;
		}
		void back_propgation(VectorXd x,VectorXd y,VectorXd *delta_b,MatrixXd *delta_w)
		{
				delta_b = new VectorXd[layers];
				delta_w = new MatrixXd[layers];
				vector<VectorXd>activation,zs;
				activation.push_back(x);
				for(int i=0;i<layers;i++) 
				{
						x=weight[i]*x+bias[i];
						zs.push_back(x);
						x=sigmoid(x);
						activation.push_back(x);
				}

				VectorXd d= cost_derivative(x,y).cwiseProduct(sigmoid_prime(zs[layers-1]));
				delta_b[layers-1] = d;
				delta_w[layers-1] = d*(x.transpose());
				for(int l=2;l<=layers;l++)
				{
						VectorXd spv = sigmoid_prime(zs[layers-l]);
						d = (weight[layers-l].transpose()*d).cwiseProduct(spv); 
						delta_b[layers-l] = d;
						delta_w[layers-l] = d*(activation[layers+1-l].transpose());
				}
		}
		VectorXd cost_derivative(VectorXd output,VectorXd y){return output-y;}
};
