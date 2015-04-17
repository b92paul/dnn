#include "vertibi.h"
#include <cstdio>
int x[] = {2,0,0};
LD **xy;
LD **yy;
int main() {
    xy = (LD**)malloc(sizeof(LD*)*3);
    xy[0] = new LD[2];
    xy[1] = new LD[2];
    xy[0][0] = 0.2;
    xy[0][1] = 0.5;
    xy[1][0] = 0.4;
    xy[1][1] = 0.4;
    xy[2][0] = 0.4;
    xy[2][1] = 0.1;
    yy = (LD**)malloc(sizeof(LD*)*2);
    yy[0] = new LD[2];
    yy[1] = new LD[2];
    yy[0][0] = 0.7;
    yy[0][1] = 0.3;
    yy[1][0] = 0.4;
    yy[1][1] = 0.6;
    Vertibi a;
    a.init(3,2,3,x,xy,yy);
    int *ret = a.work();
    for(int i=0;i<3;i++)printf("%d ",ret[i]);
    puts("");
    return 0;
}