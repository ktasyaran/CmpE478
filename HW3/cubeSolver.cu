#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

//Required constants.
#define EPSILON 1E-4 
#define MAXITER 7000



//Useful functions for finding partial derivatives and exact solution(assumption for existence of the solution).
float exact(float x,float y,float z)
{
  return(x*x+y*y+z*z) ; 
}
//Partial derivation function with respect to uxx, uyy and uzz.
float f(float x,float y,float z) 
{
  return(6.0) ; 
}

/*
Saxpy functor to solve linear differential equation.
*/
struct saxpy_functor: public thrust::binary_function<float,float,float>
{
	 int prev;
	 int cur;
	const int N;
	float * ptr;
  //Constructor takes previous index, current index, N and a vector which holds all points.
	saxpy_functor(int _prev,int _cur,int _N,float *_ptr):prev(_prev),cur(_cur),N(_N),ptr(_ptr){}

    //Jacobi iteration is done here.
	__host__ __device__
		float operator()( const thrust::tuple<int,int,int> &y,const float &x) const{
			int a,b,c;
			a=thrust::get<0>(y);
			b=thrust::get<1>(y);
			c=thrust::get<2>(y);
			float h=1.0/(N-1);
			ptr[2*(a*N*N+b*N+c)+cur]=(1.0/6)*(ptr[2*((a-1)*N*N+b*N+c)+prev]+ptr[2*((a+1)*N*N+b*N+c)+prev]+ptr[2*(a*N*N+(b+1)*N+c)+prev]+ptr[2*(a*N*N+(b-1)*N+c)+prev]+ptr[2*(a*N*N+b*N+c+1)+prev]+ptr[2*(a*N*N+b*N+c-1)+prev])-(h*h);
  			float diff=ptr[2*(a*N*N+b*N+c)+cur]-ptr[2*(a*N*N+b*N+c)+prev];
			return diff*diff;
	}

};
 int i,j,k ; 
using namespace std;


int main(int argc, char**argv)
{
  float h,x,y,z ; 
  if(argc!=2){
  	cout<<"Usage: ./executable N";
  	exit(-1);
  }
  
  int N;
  N=atoi(argv[1]);
  h = 1.0/N ; 
  //Vector initialization to hold all numbers. It will be a 3D vector. (It will be flattened later.)
  //These variables are multiplied by 2 because flattened vector will hold both previous and current values.
  unsigned long long int start_index=2*((N+1)*(N+1) + (N+1) + 1);
  unsigned long long int end_index=2*((N+1)*(N+1)*(N-1) + (N+1)*(N-1) + (N-1));

  int dims[4]={N+1,N+1,N+1,2};
  
  //Total dimension required for 1D array.
  unsigned long long int total_dims=dims[0]*dims[1]*dims[2]*dims[3];
  //Holds all numbers. Flattened version of 3D vector.
  thrust::host_vector <float> nums(total_dims);

  // Assigning non-boundary points for all 3 dimensions.
  for(i=start_index;i<=end_index;i++){
  	nums[i]=0.0;
  }

  // Assigning boundary points for dimension X.
  for(i=0 ; i <= N ; i++) {
    x = i*h ;
    nums[2*(i*dims[0]*dims[1])]=exact(x,0.0,0.0);
    nums[2*(i*dims[0]*dims[1]+N*dims[0]+N)]=exact(x,1.0,1.0);
  }
  // Assigning boundary points for dimension Y.
  for(j=0 ; j <= N ; j++) {
    y = j*h ; 
     nums[2*j*dims[0]] =exact(0.0,y,0.0);
    nums[2*(N*dims[0]*dims[1]+j*dims[0]+N)]=exact(1.0,y,1.0);

}
// Assigning boundary points for dimension Z.
 for(k=0 ; k <= N ; k++) {
 	z=k*h;
     nums[2*k] =exact(0.0,0.0,z);
    nums[2*(N*dims[0]*dims[1]+N*dims[0]+k)]=exact(1.0,1.0,z);     
}
	
	//Transfer from host to device vector.
	thrust::device_vector <float> Dnums=nums;
	thrust::device_vector <float> result((N-1)*(N-1)*(N-1));
	thrust::device_vector<thrust::tuple<float,float,float> > iterations;
	for(i=1 ; i < N ; i++) {
      for(j=1 ; j < N ; j++) {
        for(k=1;k<N;k++){
        	iterations.push_back(thrust::make_tuple(i,j,k));
        }
    }
}

	

	// iteration loop until convergence.
	
 unsigned int iter,prev,cur ; 
 float sum;
 iter = 0 ; 
  prev=0;
  sum = 1.0E30 ; 
  //Start time of parallel execution
const clock_t begin_time = clock();
//Solves differential equation until sum is greater than epsilon
  while( (sum > EPSILON)   && (iter < MAXITER) ) {
     cur = (prev + 1) % 2 ; 
   	sum=0.0;
    //Transformation is required to solve linear system. Saxpy does Jacobi iteration.
   	thrust::transform(iterations.begin(),iterations.end(),Dnums.begin(),result.begin(),saxpy_functor(prev,cur,N+1,thrust::raw_pointer_cast(&(Dnums[0]))));
  	float tot=thrust::reduce(result.begin(),result.end(),(float)0,thrust::plus<float>());
    //Result is reduced and added.
    sum+=tot;
    iter = iter + 1 ;
    prev = cur ; 
   // printf("%f\n",sum);
  }
//End time of parallel execution
  printf("%d\t%d\t%.5f\n",N,iter,float( clock () - begin_time ) /  CLOCKS_PER_SEC);
  	
  	
   
	
 	

 
  return 0;
}