/*
 * =====================================================================================
 *
 *       Filename:  implus_x_y.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年06月01日 16时57分11秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  NDY (sogou-ime-research), nidaye@sogou-inc.com
 *        Company:  sogou-inc
 *
 * =====================================================================================
 */

/************************************************************************
	> File Name: implus.cpp
	> Author:  implus for speed up
	> Mail:    implusdream@gmail.com
	> Created Time: Tue 29 Mar 2016 05:14:58 AM PDT
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
using namespace std;

vector<vector<double> > vvdx, vvdy;
struct node {
    int x, y;
    double val;
    node(){}
    node(int x, int y, double val):x(x), y(y), val(val){}
    bool operator<(const node& ths) const {
        return val < ths.val;
    }
};
vector<vector<node> > probx, proby;
vector<vector<int> >  element_array;
struct qnode{
    int word, id, cnt;
    double sum, key;
    qnode(){}
    qnode(int word, int id, int cnt, double sum, double key):word(word), id(id), cnt(cnt), sum(sum), key(key){}
    bool operator<(const qnode& ths) const{
        return key < ths.key;
    }
};
priority_queue<qnode> pq;

const int MAX_LEN = 12345678;
char buf[MAX_LEN];

void file2vvd(char* filename, vector<vector<double> >& vvd){
    fstream f(filename, ios::in);
    while(f.getline(buf, MAX_LEN)){
        stringstream ss(buf); double val;
        vvd.push_back(vector<double>());
        while(ss >> val){
            vvd[vvd.size() - 1].push_back(val);
        }
    }
    f.close();
}
void element_array2file(char* filename){
    cerr<<filename<<" file to be saved"<<endl;
    fstream f(filename, ios::out);
    for(int i = 0; i < element_array.size(); i++){
        for(int j = 0; j < element_array[i].size(); j++){
            f << element_array[i][j] + 1 << " ";
        }
        f << "\n";
    }
    f.close();
}


vector<string> idx2word;
void idx2word2idx2word(char * idx2word_file){
    fstream f(idx2word_file, ios::in);
    cerr<<idx2word_file<<" idx2word load..."<<endl;
    string str;
    while(f >> str){
        idx2word.push_back(str);
    }
}

void element_array2text(const char* filename){
    cerr<<filename<<" text string file to be saved"<<endl;
    fstream f(filename, ios::out);
    for(int i = 0; i < element_array.size(); i++){
        for(int j = 0; j < element_array[i].size(); j++){
            int v = element_array[i][j];
            if(v == -1){
                f << "<null>" <<" ";
            }else{
                f<< idx2word[v]<<" ";
            }
        }
        f <<"\n";
    }
    f.close();
}

int main(int argc, char* argv[]){
    ios_base::sync_with_stdio(false);
    // 1, 2, 3, 4
    idx2word2idx2word(argv[4]);
    cerr<<" read files into vvdx vvdy"<<endl;
    file2vvd(argv[1], vvdx);
    file2vvd(argv[2], vvdy);

    int vocab_size = vvdx.size();
    int vocab_sqrt = vvdx[0].size();
    cerr<<"vocab_size = "<<vocab_size<<endl;
    cerr<<"vocab_sqrt = "<<vocab_sqrt<<endl;

    //generate probx
    cerr<<"generate probx"<<endl;
    for(int word = 0; word < vocab_size; word++){
        probx.push_back(vector<node>());
        proby.push_back(vector<node>());
        for(int x = 0; x < vocab_sqrt; x++){
            //for(int y = 0; y < vocab_sqrt; y++){
                //node d(x, y, vvdx[word][x] + vvdy[word][y]);
                node d(x, x, vvdx[word][x]);
                probx[word].push_back(d);
                node dd(x, x, vvdy[word][x]);
                proby[word].push_back(dd);
            //}
        }
        sort(probx[word].begin(), probx[word].end());
        sort(proby[word].begin(), proby[word].end());
        if(word % 1000 == 0) cerr<<" word = "<<word<<" finished!"<<endl;
    }

    cerr<<"probx size = "<<probx.size()<<","<<probx[0].size()<<endl;
    cerr<<"proby size = "<<proby.size()<<","<<proby[0].size()<<endl;
    for(int i = 0; i < probx.size(); i++){
        vector<node>& vn = probx[i];
        double sum = 0;
        for(int j = 1; j < vn.size(); j++){
            sum += vn[j].val;
        }
        qnode one(i, 0, vn.size() - 1, sum, 0);
        one.key = one.sum/one.cnt - probx[one.word][one.id].val;
        pq.push(one);
    }

    for(int i = 0; i < vocab_sqrt; i++){
        //element_array.push_back(vector<int>(vocab_sqrt, -1));
        element_array.push_back(vector<int>());
    }

    int finished = 0;
    cerr<<"queue x begin!"<< pq.size() <<endl;
    while( pq.size() > 0 ) {
        qnode one = pq.top(); pq.pop();
        node  pos = probx[one.word][one.id];
        if(element_array[pos.x].size() >= vocab_sqrt){
            one.cnt -= 1;
            one.sum -= pos.val;
            one.id  += 1;
            one.key = one.sum/one.cnt - probx[one.word][one.id].val;
            pq.push(one);
        }else{
            element_array[pos.x].push_back(one.word); ++finished;
            if(one.id == 0){
                sprintf(buf, "for x dim, %30s\t locate in (%5d, xxx) one.id = %5d; finished = %10d\n", idx2word[one.word].c_str(), pos.x, one.id, finished);
                cerr<<buf;
            }
        }
    }
//  above fixed x dim, then adjust y dim
    finished = 0;
    for(int i = 0; i < vocab_sqrt; i++){
        assert(pq.size() == 0);
        assert(element_array[i].size() <= vocab_sqrt);
        for(int j = 0; j < element_array[i].size(); j++){
            int w = element_array[i][j];
            // here we got proby
            vector<node>& vn = proby[w];
            double sum = 0;
            for(int k = 1; k < vn.size(); k++) sum += vn[k].val;
    //qnode(int word, int id, int cnt, double sum, double key):word(word), id(id), cnt(cnt), sum(sum), key(key){}
            qnode one(w, 0, vn.size() - 1, sum, 0);
            one.key = one.sum/one.cnt - proby[one.word][one.id].val;
            pq.push(one);
            element_array[i][j] = -1; // clear and adjust
        }
        // if element_array[i] is less than vocab_sqrt, add -1
        for(int add = element_array[i].size(); add < vocab_sqrt; add++){
            element_array[i].push_back(-1);
        }
        cerr<<i<<" queue y begin!"<< pq.size() << endl;
        while(pq.size() > 0){
            qnode one = pq.top(); pq.pop();
            node pos = proby[one.word][one.id];
            if(element_array[i][pos.y] >= 0){
                one.cnt -= 1;
                one.sum -= pos.val;
                one.id  += 1;
                one.key  = one.sum/one.cnt - proby[one.word][one.id].val;
                pq.push(one);
            }else{
                element_array[i][pos.y] = one.word; ++finished;
                if(one.id == 0){
                    sprintf(buf, "for y dim, %30s\t locate in (%5d, %5d) one.id = %5d; finished = %10d\n", idx2word[one.word].c_str(), i, pos.y, one.id, finished);
                    cerr<<buf;
                }
            }
        }
    }
    element_array2file(argv[3]);
    string tmpfile = argv[3]; tmpfile += ".dic_string";tmpfile += argv[5];
		cout<<"tempfile"<<tmpfile<<endl;
    element_array2text(tmpfile.c_str());

    return 0;
}

