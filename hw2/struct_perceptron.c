#include<stdio.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
#include"vertibi.h"
#include<time.h>
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
      }
    }
    examples[i].x.feature = array;
    // read y
    examples[i].y.phone = (int*) malloc(frame*sizeof(int));
    examples[i].y.frame = frame;
    for (j = 0; j < frame; ++j)
      scanf("%d", &examples[i].y.phone[j]);
  }
  
  puts("Data read.");
  sample.n=n;
  sample.examples=examples;
  return sample;
}
SAMPLE read_test(char *file)
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
      }
    }
    examples[i].x.feature = array;
  }
  puts("Test read.");
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
void calc_psi(double *w,PATTERN x,LABEL y,int flag)
{	
  long sizePsi = 69*48+48*48;
  int i,j;
  long frame = x.frame;
  long length = x.length; // 69
  for(i=0;i<sizePsi;i++) my_psi[i]=0;
  for(i=0;i<frame;i++) {
    long label = y.phone[i];
    for(j=0;j<length;j++) {
      my_psi[label*69+j] += x.feature[i][j]; 
    }
  }
  for(i=1;i<frame;i++) {
    long label1 = y.phone[i-1]; // 0~47
    long label2 = y.phone[i];
    my_psi[69*48+label1*48+label2]+=1.0;
  }
	if(flag==1) for(i=1;i<=sizePsi;i++) w[i]+=my_psi[i-1];
	else if(flag==2) for(i=1;i<=sizePsi;i++) w[i]-=my_psi[i-1];
}
/*double dot(double *w,PATTERN x,LABEL y)
{
  long sizePsi = 69*48+48*48;
  int i,j;
  long frame = x.frame;
  long length = x.length; // 69
  for(i=0;i<sizePsi;i++) my_psi[i]=0;
  for(i=0;i<frame;i++) {
    long label = y.phone[i];
    for(j=0;j<length;j++) {
      my_psi[label*69+j] += x.feature[i][j]; 
    }
  }
  for(i=1;i<frame;i++) {
    long label1 = y.phone[i-1]; // 0~47
    long label2 = y.phone[i];
    my_psi[69*48+label1*48+label2]+=1.0;
  }
	double ans=0;
	for(i=1;i<=sizePsi;i++) ans+=w[i]*my_psi[i-1];
	return ans;
}*/
int main(int argc,char **argv)
{
	if(argc<3) 
	{
		printf("argv: training data,testing data,iteration\n");
		return 0;
	}
	SAMPLE sample = read_training(argv[1]);
	double *weight;
	int w_len = 69*48+48*48+1,i,j;
	weight = (double*)malloc(sizeof(double)*w_len);
	for(i=0;i<w_len;i++) weight[i]=0;
	int is_break;
	int iteration = 100000,T=0;
	if(argc>=4) iteration = atoi(argv[3]);
	LABEL yhat;
	int loss_value,total;
  int sizePsi = 69*48+48*48;
	my_psi = (double*)malloc((sizePsi+1)*sizeof(double));
	while(T<iteration)
	{
		T=T+1;
		total=0;
		for(i=0;i<sample.n;i++)
		{
  		yhat.frame = sample.examples[i].x.frame;
  		yhat.phone = work_vertibi_loss_psi(sample.examples[i].x, 48,weight , NULL);
			loss_value=loss_func(yhat,sample.examples[i].y);
			if(loss_value!=0)
			{
				total+=loss_value;
				calc_psi(weight,sample.examples[i].x,sample.examples[i].y,1);
				calc_psi(weight,sample.examples[i].x,yhat,2);//w+=psi-psi	
			}
			free(yhat.phone);
		}
		printf("%d:%d\n",T,total);
	}
	//predict
	SAMPLE test = read_test(argv[2]);
	FILE *fp = fopen("perceptron.out","w");
	for(i=0;i<test.n;i++)
	{		
		LABEL yhat;
  	yhat.frame = test.examples[i].x.frame;
  	yhat.phone = work_vertibi_loss_psi(test.examples[i].x, 48,weight , NULL);
		for(j=0;j<yhat.frame;j++)
    	fprintf(fp, "%d%c", yhat.phone[j], (j == yhat.frame - 1)?'\n':' ');
	}
	free(my_psi);
	free(weight);
	return 0;
}
