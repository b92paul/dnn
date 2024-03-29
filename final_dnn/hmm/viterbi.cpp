#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
using namespace std;
const int N = 2000;
const int state = 1943;
int sen_num=0;

char name_list[1000][100];
int len_list[1000];
double yy_table[N][N];

double prob[N];

int table[N][N] = {};
double DP[2][N];
int arg_max(double list[],int len){
  int midx = -1;
  double res = -1e30;
  for(int i=0;i<len;i++){
    if(list[i]>res){
      midx = i;
      res = list[i];
    }
  }
  assert(midx!=-1);
  return midx;
}
string probs_path = "hmm/probs/";
string out_path = "hmm/out/";
int main(int argc,char* argv[]){
  if(argc!=3){
    puts("input probs name, out name!!");
    return 1;
  } else{
    probs_path += argv[1];
    out_path += argv[2];
  }
  FILE* xy_file = fopen(probs_path.c_str(),"r");
  if(xy_file==NULL){
    puts("no such file!!");
    return 1;
  }
  FILE* yy_file = fopen("hmm/hmm_table.out","r");
  FILE* name_file = fopen("hmm/sentence_name_len.out","r");
  int size;
  char name[100];
  for(int i=0;i<state;i++){
    for(int j=0;j<state;j++)fscanf(yy_file,"%lf",&yy_table[i][j]);
  }
  puts("read yy table done!");
  fclose(yy_file);
  FILE* fout = fopen(out_path.c_str(),"w");
  while(~fscanf(name_file,"%s %d",name, &size)){

    printf("%s %d\n",name,size);
    sen_num++;
    int f =0 ;
    for(int i=0; i<size;i++){
      for(int j=0;j<state;j++){
        fscanf(xy_file,"%lf",&prob[j]);
      }
      if(i==0){
        for(int j=0;j<1943;j++) DP[f][j] = prob[j];
        f = 1-f;
      } else{
        for(int j=0;j<1943;j++){
          DP[f][j] = -1e100;
          int midx = -1;
          for(int k=0;k<1943;k++){
            if(DP[1-f][k] +yy_table[k][j] > DP[f][j]){
              midx = k;
              DP[f][j] = DP[1-f][k]+yy_table[k][j];
            }
          }
          assert(midx!=-1);
          DP[f][j] += prob[j];
          table[i][j] = midx;
        }
        f = 1-f;
      }
    }
    vector<int> tmp;
    int end = arg_max(DP[1-f],size);
    for(int i=size-1;i>=0;i--){
      tmp.push_back(end);
      end = table[i][end];
    }
    assert(end==0);
      fprintf(fout,"%s",name);
    for(int i=size-1;i>=0;i--){
      fprintf(fout," %d",tmp[i]);
    }
    fprintf(fout,"\n");
    //if(sen_num==3)break;
  }
  fclose(fout);
  fclose(name_file);
  fclose(xy_file);
}
