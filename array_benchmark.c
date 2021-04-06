#include <stdio.h>

int main () {

   /* an array with 5 rows and 2 columns*/
   double arr[5428][5428];
   int i, j;

   /* output each array element's value */
   for ( i = 0; i < 5; i++ ) {
      for ( j = 0; j < 2; j++ ) {
         arr[i][j] = i*i + j*j;
      }
   }

   return 0;
}