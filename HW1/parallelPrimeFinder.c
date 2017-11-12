#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <pwd.h>
#include <string.h>
#include <time.h>

//Output file name.
#define FILENAME "/omp_result.csv"
//Ordered prime numbers output file.
#define ORDERED_PRIMES "/tmp/ordered.txt"
//Output chunk for writing newline to file.
#define OUT_CHUNK 20
//Number of primes predefined for larger n to create prime array.
#define PRIME_SIZE 1000000000
//First thread number for sequential. It will be increased in loop via right shifting.
#define FIRST_THREAD_NUM 1
//Default chunk size.
#define CHUNK_SIZE 7000
//Max thread number for machine i.e (1, 2, 4, 8)
#define MAX_LIM 4
//scheduler types for openmp (dynamic, static and guided. auto is ignored although it exists.)
#define MAX_SCH 3

//Some initialization.
unsigned long long int i,j,newones,firstused;
#pragma omp threadprivate(i,j)
char* homedir;
char* schedulers[3]={"static","dynami","guided"};
unsigned long long int primes[PRIME_SIZE];

/*
Gcd algorithm to check primality of numbers in sequential part. Runs O(log max{a,b}) time.
*/

int gcd(unsigned long long int a, unsigned long long int b){
	unsigned long long int r;
	while(b!=0){
		r=a%b;
		a=b;
		b=r;
	}
	return a;
}

/*
Prime number finder up to number N using openMP N will be very large.
*/

int main(int argc, char**argv){
	unsigned long long int N,mySQT,ort,ort1;
	int k,s;
	firstused=0;
	FILE * fp=NULL;
	/*
		This part also includes some definition.
	*/
	if((homedir=getenv("HOME"))==NULL){
		homedir=getpwuid(getuid())->pw_dir;
	}
	fp=fopen(strcat(homedir,FILENAME),"w");
    fprintf(fp,"%s\t\t%s\t%s\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n","M","Scheduler","Chunk","T1","T2","T4","T8","S2","S4","S8");
	N=atoi(argv[1]);
	primes[0]=2;
	firstused++;

	mySQT=round(sqrt(N));
	int chunk=CHUNK_SIZE;
	omp_lock_t writelock;
	//Mutex lock initialization.
	omp_init_lock(&writelock);
	unsigned long long int my_gcd,rem;
	// To get time difference from start to end for sequential case (from 3 to sqrt(N)).
	 double seq_start,seq_end;
     double diff_t;
	/*
		Sequential part is for getting prime numbers from 3 to sqrt(N).
		The first number is already initialized as 2.
	*/

    //Base time for sequential generation. It will run once. This base time will be added to others for calculating speedup.
 
  seq_start = omp_get_wtime() ; 
	for(i=3;i<=mySQT;i+=2){
		for(j=0;j<firstused;j++){
			my_gcd=gcd(i,primes[j]);
			if(my_gcd!=1){	
				break;
			}
		}
		if(j==firstused){
				primes[firstused]=i;
				firstused++;
		}
	}

	seq_end = omp_get_wtime() ; 
	//Base difference time for sequential start and sequential end.
	diff_t=seq_end-seq_start;
	//Setting first iteration for sqrt(N).
	mySQT=mySQT+((mySQT & 0x1)==0 ? 1:2);
	
	//tid=omp_get_thread_num();
	/*
		This is parallel part for finding primes from sqrt(N)+1 (if sqrt(N) is even) or sqrt(N)+2 (if sqrt(N) is odd)
		to N with incrementing by 2 (Since there are no even prime numbers other than 2) in parallel.
		N also can be prime itself.
	*/
	/*
	This is scheduler setter loop. 3 Scheduler type is used: Static, dynamic and guided. Auto is ignored because auto 
	ignores chunk size.
	*/
		for(k=0;k<MAX_SCH;k++){
			
			fprintf(fp,"%lld\t%s\t%d\t",N,schedulers[k],CHUNK_SIZE);
		/*
			Schedule setter part. Depends on k, respective schedule will set with chunk size.
		*/
		if(k==0){
			omp_set_schedule(omp_sched_static,chunk);
		}else if(k==1){
			omp_set_schedule(omp_sched_dynamic,chunk);
		}else if(k==2){
			omp_set_schedule(omp_sched_guided,chunk);
		}
		double times[4];
		times[0]=times[1]=times[2]=times[3]=diff_t;
		//This loop is for setting number of threads.
		for(s=0;s<MAX_LIM;s++){
		//Times for parallel start and parallel end.
		double par_start, par_end;
		
		newones=firstused;
		par_start = omp_get_wtime(); 
		
		#pragma omp parallel shared(primes,firstused,N,newones) num_threads(FIRST_THREAD_NUM<<s) private(rem)
		{	
		
			#pragma omp for 
			for(i=mySQT;i<=N;i+=2){
				/*Since there are no even numbers in iterations, dividing to 2 is needless. Therefore, division and
				modulo comparison will start from 3 (firstused[1]).
				*/
				for(j=1;j<firstused;j++){
					//Private remainder to check modulo.
					rem=i%primes[j];
					if(rem==0){
					
						j=firstused*2;
					}	
				}
			
				if(j==firstused){
				//mutex lock for exclusive write. Inside this lock and unlock statements is critical section.
				omp_set_lock(&writelock);
				primes[newones]=i;
				newones++;
				omp_unset_lock(&writelock);
				}
			
			}
		}
		par_end = omp_get_wtime() ; 
		times[s]+=(par_end-par_start);
		
	}
	//writes result into file.
	fprintf(fp,"%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\n",times[0],times[1],times[2],times[3],(times[0]/times[1]),(times[0]/times[2]),(times[0]/times[3]));
}
	//lock destruction.
	omp_destroy_lock(&writelock);
	//Closes output file.
	fclose(fp);
	fp=fopen(ORDERED_PRIMES,"w");
	/*
		Bubble sort algorithm to sort the big prime numbers array (the used part of course).
	*/
	fprintf(fp, "%lld\n",newones);
	for(ort=0;ort<newones-1;ort++){
		for(ort1=0;ort1<newones-1-ort;ort1++){
			if(primes[ort1]>primes[ort1+1]){
				primes[ort1]=primes[ort1]^primes[ort1+1];
				primes[ort1+1]=primes[ort1]^primes[ort1+1];
				primes[ort1]=primes[ort1]^primes[ort1+1];
			}
		}
	}
	/*
		Prints output to file in chunks of given chunk size.
	*/
	for(ort=0;ort<newones;ort++){
		
		if(ort%OUT_CHUNK==0 && ort>0){
			fprintf(fp, "%lld\n",primes[ort]);
		}else{
			fprintf(fp, "%lld ",primes[ort]);
		}
	}
	fclose(fp);
	return 0;
}
