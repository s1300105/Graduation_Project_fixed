#define SQ(x) ((x)*(x))

int main()
{
    int n;
    scanf("%d", &n);
    printf("%d\n", SQ(n) + SQ(n + 1));
    return 0;
}
