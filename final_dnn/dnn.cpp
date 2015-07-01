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
//color for print
char color[]="\033[0;32m";
char NC[]="\033[0m";

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
inline void flogistic(const MatrixXd& z, MatrixXd& a){
  //a = ((z*0.4).array().max(0).min(1));
  a = ((-z).array().exp()+1).array().inverse();
  //a = ((z.array() <=2.5 && z.array()>=-2.5).select(z*0.4,1.0));
  return;
}

void s2p(const MatrixXd& x, MatrixXd& out){
  //out = ( (x.array()<0 || x.array()>1).select(MatrixXd::Zero(x.rows(),x.cols()), 0.4));
  out = (1-x.array())*(x.array());
  return;
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
            bias[i] = RandomVet(neuron[i]);
            //bias[i] = VectorXd::Random(neuron[i]) ;
            int num;
            if(i==0) num=input_size;
            else num=neuron[i-1];
            weight[i] = RandomMat(neuron[i],num)/ sqrt((double)num);
            //weight[i] = MatrixXd::Random(neuron[i],num)/ sqrt((double)num) *3; //sigma -1 ~ 1
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
    
    MatrixXd fast_feedforward(MatrixXd x)
    {
        for(int i=0;i<layers;i++) flogistic(weight[i]*x+bias[i]*(VectorXd::Ones(x.cols()).T()),x);
        return x;
    }
    void fast_back_propagation(const MatrixXd& x,const MatrixXd& y,VectorXd *delta_b,MatrixXd *delta_w)
    {
        activation[0]=x;
        for(int i=0;i<layers;i++) {
            zs[i]=weight[i]*activation[i]+bias[i]* (VectorXd::Ones(x.cols()).T()) ;
            flogistic(zs[i], activation[i+1]);
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

    
    void cost_derivative(const MatrixXd& a,const MatrixXd& y,MatrixXd& output){
      output=a;
      #pragma omp parallel for
      for(int i=0;i<y.cols();i++){
        output(int(y(0,i)),i) -=1.0;
      }
      return;
    }

    void SGD(MatrixXd& BX, MatrixXd& BY, double eta, int epochs, int msize, bool decay, double T, 
        MatrixXd& ValX, MatrixXd& ValY ,MatrixXd& testX, 
        bool findModel = false, vector<int>* param = NULL, // for model_finder
        vector<pair<double,double> >* ans = NULL){
      int count =0 ,end=msize;
      puts("-- Start SGD.");
      for(int l=1;l<= layers;l++){
        activation[l] = MatrixXd(neuron[l-1],msize);
        delta[l-1] = MatrixXd(neuron[l-1],msize);
      }
      clock_t start_time = clock();
      struct timeval tstart, tend;
      gettimeofday(&tstart, NULL);
      
      for(int i=0; i<epochs; i++){
        if(end > BX.cols()){
          count=0,end=msize;
          //shuffleTrain(BX, BY);
        }
        
        int num = 2000;
        if(findModel){
          num = (*param)[0];
          if(i == (*param)[1])break;
        }
        //update by back propagation
        update(BX.block(0,count,BX.rows(),msize),BY.block(0,count,BY.rows(),msize),eta,i, T, decay);
        
        if((i+1) % num == 0){
          e_val = fast_eval(ValX,ValY);
          e_in = fast_eval(BX.block(0,count,BX.rows() ,msize),BY.block(0,count,BY.rows(),msize));
          // print exp message
          gettimeofday(&tend, NULL);
          double time_delta = ((tend.tv_sec  - tstart.tv_sec) * 1000000u + tend.tv_usec - tstart.tv_usec) / 1.e6;
          printf("%s-- Spend %f time to train %d batch.\n",color,(time_delta),num);
          printf("Layer number = %d; ",layers);
          for(int i=0;i<layers;i++)printf("%d%c",neuron[i],i==(layers-1)?'\n':',');
          printf("learning rate = %.3f, momentum = %.3f\n",eta, momentum);
          printf("learning rate now = %.3f\n",(decay)?(eta/(i/T + 1)):eta);
          printf("e_val = %lf\n",e_val);
          printf("e_in of batch = %lf\n",e_in);
          printf("-- batch %d done.%s\n",i+1,NC);
          //return parameter for model finder
          if(findModel)ans->push_back(make_pair(e_val,e_in));
          start_time = clock(); 
          gettimeofday(&tstart, NULL);
        }
        count+=msize, end+=msize;
        if( (i+1)% num * 2 == 0){
          char model_name[100]="";
          sprintf(model_name,"model_%.4lf_%.4lf.out", e_in, e_val);
          puts(model_name);
          save_model(model_name);
          //Predict(testX);
        }
      }
    }
    void update(Block<MatrixXd> BX, Block<MatrixXd> BY,double eta,int time, double T, bool decay){
        double msize = (double)BX.cols();
        /*
        if(e_val>=0.55) eta/=8;
        else if(e_val>=0.53) eta/=4;
        else if(e_val>=0.5) eta/=2;
        */
        if(decay) eta = eta /(time/T +1);
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
    
    double fast_eval(const MatrixXd& ValBatchX, const MatrixXd& ValBatchY)
    {
        double num=0;
        MatrixXd tmp = fast_feedforward(ValBatchX);
        #pragma omp parallel for
        for(int i=0;i< tmp.cols();i++)
            if(max_number(tmp.col(i)) == (int)ValBatchY(0,i)) num++;
        return num/ValBatchX.cols();
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
