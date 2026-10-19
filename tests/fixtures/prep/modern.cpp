#include <bits/stdc++.h>
using namespace std;
#define rep(i, n) for (int i = 0; i < (n); i++)
using ll = long long;
template <class T> T sq(T x) { return x * x; }
int main() {
    ll n; cin >> n;
    vector<ll> v(n);
    rep(i, n) cin >> v[i];
    cout << sq(v[0]) << "\n";
}
