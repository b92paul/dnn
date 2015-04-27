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
}Vertibi;

void free2(void **c,int len) {
    int i;
    if(c) {
        FOR(i,len)free(c[i]);
        free(c);
        c=0;
    }
}
void copy_int(int **t, int *s, int len) {
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

void copy_ld2(LD ***tar, LD **src, int x,int y){
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
void validation_x(Vertibi *v) {
    int i;
    FOR(i,v->len){
        assert(0<=v->x[i] && v->x[i]<v->x_num);
    }
}
void init_vertibi(Vertibi *v, int _x_num, int _y_num,int _len, int *_x, LD **_xy, LD **_yy) {
    v->x_num = _x_num;
    v->y_num = _y_num;
    v->len = _len;
    copy_int(&(v->x), _x, _len);
    copy_ld2(&(v->xy), _xy, _x_num, _y_num);
    copy_ld2(&(v->yy), _yy, _y_num, _y_num);
    validation_x(v);
}
int* work_vertibi(Vertibi *v) {
    int i,j,k;
    int len=v->len,x_num=v->x_num,y_num=v->y_num;
    int **par=0;
    LD **dp=0;
    copy_int2(&par,NULL, len,y_num);
    copy_ld2(&dp,NULL,len,y_num);
    FOR(j,y_num)dp[0][j] = v->xy[v->x[0]][j];
    for(i=1;i<len;i++) {
        FOR(j,y_num) {
            par[i][j] = -1;
            dp[i][j] = -MAX;
            FOR(k,y_num) {
                LD tmp = dp[i-1][k]*v->yy[k][j];
                if(tmp > dp[i][j]){
                    dp[i][j] = tmp;
                    par[i][j] = k;
                }
            }
            dp[i][j] *= v->xy[v->x[i]][j];
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
void clear_vertibi(Vertibi *v) {
    free2((void**)v->xy, v->x_num);
    free2((void**)v->yy, v->y_num);
}
#endif