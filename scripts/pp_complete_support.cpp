// Exhaustive existence of 1/2/3-withdrawal decompositions, grouped by amount.
// Output is the minimum witness size for each deposit/withdrawal pair (0=none).
// No witness cap: multiplicity is checked, but witnesses are not counted.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <vector>
using namespace std;
struct Event { double t; int64_t a; };
int main(int argc, char** argv) {
  if(argc!=3 && argc!=4) return 2;
  // Optional fourth argument: post-deposit horizon in days; return a bitmask
  // of ALL supported sizes, not just the smallest size. Legacy mode unchanged.
  const bool bits=argc==4;
  const double horizon=bits ? stod(argv[3]) : numeric_limits<double>::infinity();
  ifstream in(argv[1]); ofstream out(argv[2], ios::binary);
  int nd,nw; int64_t tol; in>>nd>>nw>>tol;
  vector<Event> d(nd),w(nw);
  for(auto& x:d) in>>x.t>>x.a;
  for(auto& x:w) in>>x.t>>x.a;
  for(const auto& dep:d) {
    map<int64_t,vector<int>> groups;
    for(int n=0;n<nw;n++) if(w[n].t>dep.t && w[n].t-dep.t<=horizon && w[n].a<=dep.a+tol)
      groups[w[n].a].push_back(n);
    vector<int64_t>a; vector<vector<int>> ids;
    for(auto& kv:groups){a.push_back(kv.first);ids.push_back(kv.second);}
    int n=a.size(); vector<uint8_t> best(n,0), result(nw,0);
    auto mark=[&](int i,int k){
      if(bits)best[i]|=1<<(k-1);
      else if(!best[i] || best[i]>k)best[i]=k;
    };
    for(int i=0;i<n;i++) if(abs(a[i]-dep.a)<=tol)mark(i,1);
    for(int i=0;i<n;i++) {
      auto lo=lower_bound(a.begin()+i,a.end(),dep.a-a[i]-tol);
      for(int j=lo-a.begin();j<n && a[i]+a[j]<=dep.a+tol;j++)
        if(i!=j || ids[i].size()>=2){mark(i,2);mark(j,2);}
    }
    for(int i=0;i<n && 3*a[i]<=dep.a+tol;i++) {
      int k=n-1;
      for(int j=i;j<n && j<=k;j++) {
        while(k>=j && a[i]+a[j]+a[k]>dep.a+tol)k--;
        for(int z=k;z>=j && a[i]+a[j]+a[z]>=dep.a-tol;z--) {
          if(i==j && j==z ? ids[i].size()<3 :
             (i==j && ids[i].size()<2)||(j==z && ids[j].size()<2)) continue;
          mark(i,3);mark(j,3);mark(z,3);
        }
      }
    }
    for(int i=0;i<n;i++) for(int idx:ids[i])result[idx]=best[i];
    out.write(reinterpret_cast<char*>(result.data()),nw);
  }
  return in && out ? 0:3;
}
