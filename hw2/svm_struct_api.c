/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/
//./svm_empty_learn  -c 1 -l 1 -v 3 ../../data/train_0.ark
#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
#include <assert.h>
//#define DEBUG_VERTIBI
#include "vertibi.h"
#define LIMIT 5000000
int min(int a, int b) {
  return a < b? a: b;
}
void init_label(LABEL *l,int frame) {
    l->frame = frame;
    l->phone = (int*)malloc(sizeof(int)*frame);
}

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  long     n;       /* number of examples */
  int i, j, k;

  puts(file);
  freopen(file, "r", stdin);
  scanf("%ld", &n);

  n = min(n, LIMIT);
  printf("n = %ld\n", n);

  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE)*n);

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
  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */

  sm->sizePsi=69*48+48*48; /* replace by appropriate number of features */
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC*)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL yhat;
  yhat.frame = x.frame;
  yhat.phone = work_vertibi_loss_psi(x, 48, sm->w, NULL);
/*  int i;
  for(i=0;i<x.frame;i++)
    printf("%d ",yhat.phone[i]);
  puts("");*/
  return yhat;
  /* insert your code for computing the predicted label y here */
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) 

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;
assert(0);
  /* insert your code for computing the label ybar here */

  return(ybar);
}
LD calc(PATTERN x, LABEL ybar, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm){
  SVECTOR *vec = psi(x,ybar,sm,sparm);
  int i=0;
  LD ret=0;
  FOR(i,sm->sizePsi)ret += vec->words[i].weight*sm->w[i];
  free(vec->words);
  ret += loss(ybar,y,sparm);
  return ret;
}
LABEL find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm) {
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)+psi(x,ybar)

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
     //printf("%lf,%lf ",sm->w[0],sm->w[1]);
  LABEL ybar;
  int i;
  ybar.frame = x.frame;
  ybar.phone = work_vertibi_loss_psi(x, 48, sm->w, &y);
  static int count = 0;
  if(0 && ++count % 100 ==0) {
    LD r1 = calc(x,ybar,y,sm,sparm);
    int c=rand()%x.frame;
    LD tmp = ybar.phone[c];
    ybar.phone[c] = rand()%48;
    LD r2 = calc(x,ybar,y,sm,sparm);
    ybar.phone[c] = tmp;
    printf("(%lf,%lf)",r1,r2);
    assert(r1 >= r2);
  }
  /*LD r1 = calc(x,ybar,y,sm,sparm); //delta(y,ybar)+w*phi(x,ybar)
  LABEL yxd;
  init_label(&yxd,x.frame);
  for(i=0;i<x.frame;i++)
    yxd.phone[i]=rand()%48;
  LD r2 = calc(x,y,yxd,sm,sparm);
  printf("(%lf %lf)",r1,r2);*/
  return ybar;
  /* insert your code for computing the label ybar here */
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return y.phone == NULL;
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  long sizePsi = sm->sizePsi;
  long i,j;
  assert(sizePsi == 69*48+48*47+48);
  WORD *words = (WORD*)malloc((sizePsi+1)*sizeof(WORD));
  long frame = x.frame;
  long length = x.length; // 69
  assert(length == 69);
  for(i=0;i<sizePsi;i++){
    words[i].wnum = i+1;
    words[i].weight=0;
  }
  words[sizePsi].wnum = 0;
  for (i = 0; i < frame; ++i) {
    long label = y.phone[i];
    for (j = 0; j < length; ++j) {
      words[label * 69 + j].weight += x.feature[i][j]; 
    }
  }
  for (i = 1; i < frame; ++i) {
    long label1 = y.phone[i - 1]; // 0~47
    long label2 = y.phone[i];
    words[69 * 48 + label1 * 48 + label2].weight += 1.0;
  }
  return create_svector_shallow(words,NULL,1.0);
  /* insert code for computing the feature vector for x and y here */
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  int i;
  assert(y.frame == ybar.frame);
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
    //assert(0);
    for(i=0;i<y.frame;i++) {
      if(y.phone[i] != ybar.phone[i])return 1;
      return 0;
    }
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
    int ret=0;
    for(i=0;i<y.frame;i++)
      if(y.phone[i] != ybar.phone[i])ret++;
    return ret;
  }
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *modelfile, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  FILE *modelfl;
  long j,i,sv_num;
  SVECTOR *v;
  MODEL *compact_model=NULL;
  MODEL *model = sm->svm_model;
 
  if(verbosity>=1) {
    printf("Writing model file..."); fflush(stdout);
  }

  /* Replace SV with single weight vector */
  if(0 && model->kernel_parm.kernel_type == LINEAR) {
    if(verbosity>=1) {
      printf("(compacting..."); fflush(stdout);
    }
    compact_model=compact_linear_model(model);
    model=compact_model;
    if(verbosity>=1) {
      printf("done)"); fflush(stdout);
    }
  }

  if ((modelfl = fopen (modelfile, "w")) == NULL)
  { perror (modelfile); exit (1); }
  fprintf(modelfl,"SVM-light Version %s\n",VERSION);
  fprintf(modelfl,"%ld # kernel type\n",
    model->kernel_parm.kernel_type);
  fprintf(modelfl,"%ld # kernel parameter -d \n",
    model->kernel_parm.poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n",
    model->kernel_parm.rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n",
    model->kernel_parm.coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n",
    model->kernel_parm.coef_const);
  fprintf(modelfl,"%s# kernel parameter -u \n",model->kernel_parm.custom);
  fprintf(modelfl,"%ld # highest feature index \n",model->totwords);
  fprintf(modelfl,"%ld # number of training documents \n",model->totdoc);
 
  sv_num=1;
  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) 
      sv_num++;
  }
  fprintf(modelfl,"%ld # number of support vectors plus 1 \n",sv_num);
  fprintf(modelfl,"%.8g # threshold b, each following line is a SV (starting with alpha*y)\n",model->b);

  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) {
      fprintf(modelfl,"%.32g ",model->alpha[i]*v->factor);
      for (j=0; (v->words[j]).wnum; j++) {
  fprintf(modelfl,"%ld:%.8g ",
    (long)(v->words[j]).wnum,
    (double)(v->words[j]).weight);
      }
      if(v->userdefined)
  fprintf(modelfl,"#%s\n",v->userdefined);
      else
  fprintf(modelfl,"#\n");
    /* NOTE: this could be made more efficient by summing the
       alpha's of identical vectors before writing them to the
       file. */
    }
  }
  fprintf(modelfl, "%ld\n", sm->sizePsi);
  for (i = 0; i < sm->sizePsi; ++i) {
    fprintf(modelfl, "%lf ", sm->w[i]);
  }
  fprintf(modelfl, "\n");
  fclose(modelfl);
  if(compact_model)
    free_model(compact_model,1);
  if(verbosity>=1) {
    printf("done\n");
  }
}

STRUCTMODEL read_struct_model(char *modelfile, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  FILE *modelfl;
  long i,queryid,slackid;
  double costfactor;
  long max_sv,max_words,ll,wpos;
  char *line,*comment;
  WORD *words;
  char version_buffer[100];
  MODEL *model;
  STRUCTMODEL sm;

  if(verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }

  nol_ll(modelfile,&max_sv,&max_words,&ll); /* scan size of model file */
  max_words+=2;
  ll+=2;

  words = (WORD *)my_malloc(sizeof(WORD)*(max_words+10));
  line = (char *)my_malloc(sizeof(char)*ll);
  model = (MODEL *)my_malloc(sizeof(MODEL));

  if ((modelfl = fopen (modelfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
  if(strcmp(version_buffer,VERSION)) {
    perror ("Version of model-file does not match version of svm_classify!"); 
    exit (1); 
  }
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);  
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
  fscanf(modelfl,"%[^#]%*[^\n]\n", model->kernel_parm.custom);

  fscanf(modelfl,"%ld%*[^\n]\n", &model->totwords);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->totdoc);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->b);

  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
  model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
  model->index=NULL;
  model->lin_weights=NULL;

  for(i=1;i<model->sv_num;i++) {
    fgets(line,(int)ll,modelfl);
    if(!parse_document(line,words,&(model->alpha[i]),&queryid,&slackid,
           &costfactor,&wpos,max_words,&comment)) {
      printf("\nParsing error while reading model file in SV %ld!\n%s",
       i,line);
      exit(1);
    }
    model->supvec[i] = create_example(-1,
              0,0,
              0.0,
              create_svector(words,comment,1.0));
  }
  free(line);
  free(words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
  //puts("wtf");
  fscanf(modelfl, "%ld", &sm.sizePsi);
  //printf("%d\n", sm.sizePsi);
  sm.w = (double*) malloc(sizeof(double)*sm.sizePsi);
  //puts("start");
  
  //printf("%d\n", sm.sizePsi);
  for (i = 0; i < sm.sizePsi; ++i) {
    //printf("i = %d\n", i);
    fscanf(modelfl, "%lf", &sm.w[i]);
  }
  sm.svm_model = model;
  //read_model(file);
  fclose(modelfl);
  return sm; 
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
  int i;
  for (i = 0 ; i < y.frame; ++i)
    fprintf(fp, "%d%c", y.phone[i], (i == y.frame - 1)?'\n':' ');
} 

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
  int i;
  for(i=0;i<x.frame;i++)
    free(x.feature[i]);
  free(x.feature);
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
  free(y.phone);
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

