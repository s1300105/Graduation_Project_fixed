#include <cstdio>

struct Point {
    int x;
    int y;
};

int dist2(struct Point a, struct Point b)
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx * dx + dy * dy;
}

int main()
{
    struct Point p = {0, 0};
    struct Point q = {3, 4};
    printf("%d\n", dist2(p, q));
    return 0;
}
