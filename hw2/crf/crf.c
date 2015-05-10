#include<stdio.h>
#include<math.h>
#include"svm_struct/svm_struct_common.h"
#include"svm_struct_api.h"
#include"vertibi.h"
#include<time.h>
#include<stdlib.h>
#define LIMIT 5000000
double *my_psi;
int min(int a,int b)
{
	if(a>b) return b;
	else return a;
}
SAMPLE read_training(char *file)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  long	n;       /* number of examples */
  int i, j, k;

  puts(file);
  freopen(file, "r", stdin);
  scanf("%ld", &n);

  n = min(n, LIMIT);
  printf("n = %ld\n", n);

  examples=(EXAMPLE *)malloc(sizeof(EXAMPLE)*n);
	double max_x=-1;
  for (i = 0; i < n; ++i) {
    if(i%500 == 0)printf("reading %d:\n", i);
    // read x
    int frame, length;
    scanf("%d%d", &frame, &length);
    scanf("%*s");
    //printf("frame = %d, length = %d\n", frame, length);
    examples[i].x.frame = frame;
    examples[i].x.length = length;
    double **array = (double**) malloc(frame*sizeof(double*));
    for (j = 0; j < frame; ++j) {
      array[j] = (double*) malloc(length*sizeof(double));
      for (k = 0; k < length; ++k) {
        scanf("%lf", &array[j][k]);
				if(array[j][k]>=max_x) max_x=array[j][k];
      }
    }
    examples[i].x.feature = array;
    // read y
    examples[i].y.phone = (int*) malloc(frame*sizeof(int));
    examples[i].y.frame = frame;
    for (j = 0; j < frame; ++j)
      scanf("%d", &examples[i].y.phone[j]);
  }
	printf("%lf\n",max_x);
  for(i=0;i<n;i++)
		for(j=0;j<examples[i].x.frame;j++)
			for(k=0;k<examples[i].x.length;k++)
				examples[i].x.feature[j][k] /= max_x;
  puts("Data read.");
  sample.n=n;
  sample.examples=examples;
  return sample;
}
int loss_func(LABEL y, LABEL ybar)
{
  int i;
  int ret=0;
  for(i=0;i<y.frame;i++)
  	if(y.phone[i] != ybar.phone[i])ret++;
  return ret;//894779
}
void calc_psi(double *w,PATTERN x,int* y,int flag,double parm)
{	
  long sizePsi = 69*48+48*48;
  int i,j;
  long frame = x.frame;
  long length = x.length; // 69
  for(i=0;i<sizePsi;i++) my_psi[i]=0;
  for(i=0;i<frame;i++) {
    long label = y[i];
    for(j=0;j<length;j++) {
      my_psi[label*69+j] += x.feature[i][j]; 
    }
  }
  for(i=1;i<frame;i++) {
    long label1 = y[i-1]; // 0~47
    long label2 = y[i];
    my_psi[69*48+label1*48+label2]+=1.0;
  }
	if(flag==1) for(i=1;i<=sizePsi;i++) w[i]+=my_psi[i-1];
	else if(flag==2) for(i=1;i<=sizePsi;i++) w[i]-=my_psi[i-1]*parm;
}
int rrr=1;
void calc_vertibi(double *update,double *weight,PATTERN x)
{
  int len=48,i;
	int **array;
	int frame = x.frame;
	double *prob;
	work_vertibi_loss_psi_48end(x,48,weight,NULL,&len,&array,&prob);
	/*double max=0;
	for(i=1;i<48;i++) 
	{
		prob[i]-=prob[0];
		if(-prob[i]>=max) max=-prob[i];
	}
	prob[0]=0;
	double sum=0;
	for(i=0;i<48;i++) 
	{
		if(max!=0) prob[i]=exp(prob[i]*10.0/max);
		if(max==0 && i==0) prob[i]=1;
		sum+=prob[i];
	}*/
	for(i=0;i<48;i++) 
	{
		//prob[i]/=sum;
		//calc_psi(update,x,array[i],2,(48-i)/1128.0);
		calc_psi(update,x,array[i],2,0.02);
		//if(rrr==205)printf("%d %lf\n",i,prob[i]);
	}
	rrr++;
	for(i=0;i<48;i++) free(array[i]);
	free(array);
	free(prob);
}
void write_pla_model(char* filename,double *weight)
{
	int w_len = 69*48+48*48+1,i;
	FILE *file=fopen(filename,"w");
	for(i=1;i<=w_len;i++)
		fprintf(file,"%lf\n",weight[i]);
	fclose(file);
	return;
}
void read_pla_model(char* filename,double *weight)
{
	int w_len = 69*48+48*48+1,i;
	FILE *file=fopen(filename,"r");
	for(i=1;i<=w_len;i++)
		fscanf(file,"%lf",&weight[i]);
	fclose(file);
	return;
}
int main(int argc,char **argv)
{
	if(argc<3) 
	{
		printf("argv: training data,testing data,iteration,batchsize,eta,read model\n");
		return 0;
	}
	SAMPLE sample = read_training(argv[1]);
	double *weight;
	double *update_w;
	int w_len = 69*48+48*48+1,i,j;
	weight = (double*)malloc(sizeof(double)*w_len);
	update_w = (double*)malloc(sizeof(double)*w_len);
	for(i=0;i<w_len;i++) 
	{
		weight[i]=0;
		update_w[i]=0;
	}
	if(argc>=7) read_pla_model(argv[6],weight);
	int iteration = 5000,T=0;
	if(argc>=4) iteration = atoi(argv[3]);
	int batch=10;
	if(argc>=5) batch = atoi(argv[4]);
	double eta=1;
	if(argc>=6) eta = atof(argv[5]);
	printf("eta=%lf\n",eta);
	LABEL yhat;
	int loss_value;
	double total;
  int sizePsi = 69*48+48*48;
	my_psi = (double*)malloc((sizePsi+1)*sizeof(double));
	int counter=0;
	FILE *record;
	record = fopen("record_aver","w");
	while(T<iteration)
	{
		T=T+1;
		total=0;
		if(T%500==0)	write_pla_model("crf.model",weight);
		for(i=0;i<sample.n;i++)
		{
  		yhat.frame = sample.examples[i].x.frame;
  		yhat.phone = work_vertibi_loss_psi(sample.examples[i].x, 48,weight , NULL);
			loss_value=loss_func(yhat,sample.examples[i].y);
			total+=(double)loss_value;
			calc_vertibi(update_w,weight,sample.examples[i].x);//w+=psi-psi	
			calc_psi(update_w,sample.examples[i].x,sample.examples[i].y.phone,1,1);
			counter++;
			if(counter == batch)
			{
				for(j=1;j<=w_len;j++) 
				{
					weight[j]+=(update_w[j]*eta);
					update_w[j]=0;
				}
				counter=0;
			}
		//	printf("%d QQ\n",i);
			free(yhat.phone);
		}
		total/=(double)sample.n;
		printf("%d:%lf\n",T,total);
		fprintf(record,"%d:%lf\n",T,total);
	}
	//predict
	SAMPLE test = read_training(argv[2]);
	FILE *fp = fopen("perceptron.out","w");
	for(i=0;i<test.n;i++)
	{		
		LABEL yhat;
  	yhat.frame = test.examples[i].x.frame;
  	yhat.phone = work_vertibi_loss_psi(test.examples[i].x, 48,weight , NULL);
		for(j=0;j<yhat.frame;j++)
    	fprintf(fp, "%d%c", yhat.phone[j], (j == yhat.frame - 1)?'\n':' ');
	}
	//free(my_psi);
	//free(weight);
	return 0;
}