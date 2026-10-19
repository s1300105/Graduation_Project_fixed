#include <stdio.h>
int a[100];
int sum(int n) { int i, s = 0; for (i = 0; i < n; i++) s += a[i]; return s; }
int main(void)
{
int n, i;
scanf("%d", &n);
for (i = 0; i < n; i++) scanf("%d", &a[i]);
printf("%d\n", sum(n));
return 0;
}
