#include<cstdio>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<fstream>
using namespace Eigen;
using namespace std;
typedef vector<VectorXd> VXd;

VectorXd sigmoid(VectorXd x)
{
		return (VectorXd((-x).array().exp())+VectorXd::Ones(x.size())).array().inverse();
}
VectorXd sigmoid_prime(VectorXd x)
{
		VectorXd sg = sigmoid(x);
		return (VectorXd::Ones(x.size())-sg).cwiseProduct(sg);
}
/*MatrixXd sigmoid(MatrixXd x)
{
		return (MatrixXd((-x).array().exp())+MatrixXd::Ones(x.size())).array().inverse();
}
MatrixXd sigmoid_prime(MatrixXd x)
{
		MatrixXd sg = sigmoid(x);
		return (MatrixXd::Ones(x.size())-sg).cwiseProduct(sg);
}*/
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
		/*void back_propgation_2(MatrixXd x,MatrixXd y,VectorXd *delta_b,MatrixXd *delta_w)
		{
				vector<MatrixXd>activation,zs;
				activation.push_back(x);
				for(int i=0;i<layers;i++) 
				{
						x=weight[i]*x+bias[i]*(VectorXd:Ones(x.cols())).transpose();
						zs.push_back(x);
						x=sigmoid(x);
						activation.push_back(x);
				}
				MatirxXd d= cost_derivative(x,y).cwiseProduct(sigmoid_prime(zs[layers-1]));
				delta_b[layers-1] += d;
				delta_w[layers-1] += d*(activation[layers-1].transpose());
				for(int l=2;l<=layers;l++)
				{
						VectorXd spv = sigmoid_prime(zs[layers-l]);
						d = (weight[layers-l+1].transpose()*d).cwiseProduct(spv); 
						delta_b[layers-l] += d;
						delta_w[layers-l] += d*(activation[layers-l].transpose());
				}
		}*/
		void back_propgation(VectorXd x,VectorXd y,VectorXd *delta_b,MatrixXd *delta_w)
		{
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
				delta_b[layers-1] += d;
				delta_w[layers-1] += d*(activation[layers-1].transpose());
				for(int l=2;l<=layers;l++)
				{
						VectorXd spv = sigmoid_prime(zs[layers-l]);
						d = (weight[layers-l+1].transpose()*d).cwiseProduct(spv); 
						delta_b[layers-l] += d;
						delta_w[layers-l] += d*(activation[layers-l].transpose());
				}
		}
		VectorXd cost_derivative(VectorXd output,VectorXd y){return output-y;}
		void SGD(VXd& TrainX, VXd& TrainY, double eta, int epochs, int msize,VXd& ValX, VXd& ValY ){
			int count =0 ;
			VXd x,y;
			VXd judge;
		//	for(int i=0; i<epochs; i++){
			int i=0;
			double z=0.2;
			while(true){
				printf("-- epoch %d start\n",i);
				if(count+msize >= TrainX.size())count=0;
				x = VXd(TrainX.begin()+count,TrainX.begin()+count+msize);
				y = VXd(TrainY.begin()+count,TrainY.begin()+count+msize);
				count+=msize;
				update(x,y,eta);
				double eva = eval(ValX,ValY);
				printf("e_val = %lf\n",eva);
				//printf("e_in of batch = %lf\n",eval(x,y));
				printf("-- epoch %d done \n",i);

				i++;
				if(eva>=z)
				{
						fstream file;
						file.open("output.QAQ",ios::out);
						for(int j=0;j<layers;j++)
						{
								file<<bias[i]<<endl;
								file<<weight[i]<<endl;
						}
						z+=0.1;
				}
			}
		}
		void update(VXd& bX, VXd& bY,double eta){
				double msize = (double)bX.size();
				VectorXd* delta_b = new VectorXd[layers];
				for(int i=0;i<layers;i++)delta_b[i] = VectorXd::Zero(neuron[i]);
				MatrixXd* delta_w = new MatrixXd[layers];
				
				for(int i=0;i<layers;i++){
					if(i==0) delta_w[i] = MatrixXd::Zero(neuron[i],input_size);
					else delta_w[i] = MatrixXd::Zero(neuron[i],neuron[i-1]);
				}
				for(int i=0;i<bX.size();i++){
					//if(i%100==0) for(int j=0;j<10;j++) printf("%lf%c",bX[i](j),j==9?'\n':' ');
					back_propgation(bX[i],bY[i],delta_b,delta_w);
				}
				for(int i=0;i<layers;i++){
					cout<<delta_b[i].maxCoeff()<<endl;
					cout<<delta_w[i].maxCoeff()<<endl;
					bias[i] -= eta*delta_b[i]/msize;
					weight[i] -= eta*delta_w[i]/msize;	
				}
		}
		double eval(VXd ValBatchX,VXd ValBatchY)
		{
				int num=0;
				int x[50]={},y[50]={};
				for(int i=0;i<ValBatchX.size();i++)
				{
						VectorXd output = feedforward(ValBatchX[i]);
						if(max_number(output) == max_number(ValBatchY[i])) num++,y[max_number(output)]++;
						x[max_number(output)]++;
				}
				for(int i=0;i<49;i++) cout<<"count "<<i<<":"<<x[i]<<" "<<y[i]<<endl;
				return (double)num/ValBatchX.size();
				/*double ans=0;
				for(int i=0;i<ValBatchX.size();i++)
				{
					double a=0;
					VectorXd output = feedforward(ValBatchX[i]);
					for(int j=0;j<output.size();j++) a+=(ValBatchY[i](j)-output(j))*(ValBatchY[i](j)-output(j));
					a=a/output.size();
					ans+=a;
				}
				ans=ans/ValBatchX.size();
				return ans;*/
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
