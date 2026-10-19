int table[4] = {1, 2, 3, 4};
int counter = 0;

int helper(int x)
{
    counter = counter + x;
    return table[x % 4] + counter;
}

int main()
{
    int i;
    int sum = 0;
    for (i = 0; i < 4; i++) {
        sum = sum + helper(i);
    }
    printf("%d\n", sum);
    return 0;
}
