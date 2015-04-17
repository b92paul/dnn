#ifndef VERTIBI_H
#define VERTIBI_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define LD double
#define FOR(i,n) for(i=0;i<n;i++)
#define MAX 1e128
typedef struct Vertibi{
    LD **xy, **yy;
    int *x;
    int len;
    int x_num,y_num;

#ifdef __cplusplus
    Vertibi(){}
    Vertibi(int _x_num, int _y_num,int _len,int *_x, LD **_xy, LD **_yy) {
        /* 
            number of x, number of y, len = days, xy=x_number*y_number, yy=y_number*y_number  
            x should has length = len 
            x should contain only 0 ~ x_num-1
        */
        init(x_num, y_num, _len, _x, _xy, _yy);
    }
    ~Vertibi() {
        clear();
    }
#endif
    void init(int _x_num, int _y_num,int _len, int *_x, LD **_xy, LD **_yy) {
        x_num = _x_num;
        y_num = _y_num;
        len = _len;
        copy(&x, _x, len);
        copy(&xy, _xy, x_num, y_num);
        copy(&yy, _yy, y_num, y_num);
        validation_x();
    }
    void clear() {
        free2((void**)xy, x_num);
        free2((void**)yy, y_num);
    }
    int* work() {
        int i,j,k;
        int **par=0;
        LD **dp=0;
        copy_int2(&par,NULL, len,y_num);
        copy(&dp,NULL,len,y_num);
        FOR(j,y_num)dp[0][j] = xy[x[0]][j];
        for(i=1;i<len;i++) {
            FOR(j,y_num) {
                par[i][j] = -1;
                dp[i][j] = -MAX;
                FOR(k,y_num) {
                    LD tmp = dp[i-1][k]*yy[k][j];
                    if(tmp > dp[i][j]){
                        dp[i][j] = tmp;
                        par[i][j] = k;
                    }
                }
                dp[i][j] *= xy[x[i]][j];
                assert(par[i][j] >=0);
            }
        }
        LD best=-MAX;
        int id = -1;
        FOR(j,y_num)
            if(dp[len-1][j] > best){
                best = dp[len-1][j];
                id = j;
            }
        int *ret = (int*)malloc(sizeof(int)*len);
        ret[len-1] = id;
        for(i=len-2;i>=0;i--) {
            id = par[i+1][id];
            ret[i] = id;
        }
        free2((void**)dp,len);
        free2((void**)par,len);
        return ret;
    }
    void free2(void **c,int len) {
        int i;
        if(c) {
            FOR(i,len)free(c[i]);
            free(c);
            c=0;
        }
    }

    void copy(int **t, int *s, int len) {
        int i;
        *t = (int*)malloc(sizeof(int)*len);
        FOR(i,len)(*t)[i]=s[i];
    }
    void copy_int2(int ***tar, int **src, int x,int y){
        int i,j;
        *tar = (int**)malloc(sizeof(int*)*x);
        FOR(i,x){
            (*tar)[i] = (int*)malloc(sizeof(int)*y);
            FOR(j,y){
                if(src)(*tar)[i][j] = src[i][j];
                else (*tar)[i][j] = 0; 
            }
        }
    }

    void copy(LD ***tar, LD **src, int x,int y){
        int i,j;
        *tar = (LD**)malloc(sizeof(LD*)*x);
        FOR(i,x){
            (*tar)[i] = (LD*)malloc(sizeof(LD)*y);
            FOR(j,y){
                if(src)(*tar)[i][j] = src[i][j];
                else (*tar)[i][j] = 0;
            }
        }
    }
    void validation_x() {
        int i;
        FOR(i,len){
            assert(0<=x[i] && x[i]<x_num);
        }
    }
}Vertibi;
#endif