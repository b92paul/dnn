#include <cstdio>
#include <Eigen/Dense>
#include <vector>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char labelPath[] = "../../data/merge/label.out";
char trainPath[] = "../../data/merge/train.out";
char testPath[]  = "../../data/merge/test.out";
char testId[]    = "../../data/merge/test_id.out";

typedef vector<double> VD;
int check = 100000;
vector<VectorXd> csvToVecters(char* filename){
	vector<VectorXd> res;
	printf("%s\n",filename);
	FILE* csv = fopen(filename, "r");
	double num;
	char split;
	vector<double> tmp;
	if(csv == NULL){
		puts("no such file !!!");
		return res;
	}
	VectorXd tmpXd;
	while(~fscanf(csv,"%lf%c",&num, &split)){
		tmp.push_back(num);
		if(split=='\n'){
			tmpXd = VectorXd::Map(&tmp[0],tmp.size());
			res.push_back(tmpXd);
			tmp.clear();
			if(res.size()%check ==0)printf("read data: %lu\n",res.size());
		}	
	}
	if(!tmp.empty()){
		tmpXd = VectorXd::Map(&tmp[0],tmp.size());
		res.push_back(tmpXd);
	}
	return res;
}

int main(){
	vector<VectorXd> res = csvToVecters(trainPath);
	printf("data size = %lu\n",res.size());
	/* try to out put	
	for(int i=0;i<res.size();i++){
		for(int j=0;j<res[i].size();j++){
			printf("%lf,",res[i][j]);
		}
		puts("");
	}
	*/
	return 0;

}
