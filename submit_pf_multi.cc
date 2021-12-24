#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <numeric>
#include <sys/time.h>
// #include "MT.h"

#define a -2
#define b -1
#define T  1

using namespace std;

double sigma_2 = pow(2, a);
double alpha_2 = pow(10, b);
// int time_count = 0;
int Max_thread_num = 0;

bool *Started;
double StartTime;
struct timeval Begin, End;
bool ClockStarted = false;

random_device seed_gen;
default_random_engine engine(seed_gen());

class Random{
private:
  vector<unsigned long> mt;
  int mti;
public:
  Random()
  {
    /* Period parameters */
    #define MT_N 624
    #define MT_M 397
    #define MATRIX_A 0x9908b0dfUL   /* constant vector a */
    #define UPPER_MASK 0x80000000UL /* most significant w-r bits */
    #define LOWER_MASK 0x7fffffffUL /* least significant r bits */

    vector<unsigned long> mt(MT_N); /* the array for the state vector  */
    int mti=MT_N+1; /* mti==MT_N+1 means mt[MT_N] is not initialized */

    this->mt = mt;
    this->mti = mti;
  }

  void init_genrand(unsigned long s)
  {
      mt[0]= s & 0xffffffffUL;
      for (mti=1; mti<MT_N; mti++)
      {
          mt[mti] =
  	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
          mt[mti] &= 0xffffffffUL;
      }
  }

  unsigned long genrand_int32(void)
  {
      unsigned long y;
      static unsigned long mag01[2]={0x0UL, MATRIX_A};

      if (mti >= MT_N)
      {
          int kk;

          if (mti == MT_N+1)
              init_genrand(5489UL);

          for (kk=0;kk<MT_N-MT_M;kk++) {
              y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
              mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
          }
          for (;kk<MT_N-1;kk++) {
              y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
              mt[kk] = mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
          }
          y = (mt[MT_N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
          mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

          mti = 0;
      }

      y = mt[mti++];

      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);

      return y;
  }

  double genrand_real3(void)
  {
      return (((double)genrand_int32()) + 0.5)*(1.0/4294967296.0);
  }

  double Uniform( void )
  {
	   return genrand_real3();
  }

  double rand_normal( double mu, double sigma )
  {
	   double z = sqrt( -2.0*log(Uniform()) ) * sin( 2.0*M_PI*Uniform() );
	   return mu + sigma*z;
 }
};

class ParticleFilter
{
private:
  int n_particle;
  double sigma_2;
  double alpha_2;
  vector< vector<double> > x;
  vector< vector<double> > x_resampled;
  vector<double> y;
  vector< vector<double> > w;
  vector< vector<double> > w_normed;
  vector<double> l;
  // vector<double> cal_time;
  int t_id;
  vector<pthread_t> tid;
  int thread_num;
  int width;

public:
  ParticleFilter(int np, double s2, double a2, int tnum)
  {
    n_particle = np;
    sigma_2 = s2;
    alpha_2 = a2;
    thread_num = tnum;
    vector<pthread_t> tid(tnum);

    vector< vector<double> > x(T+1, vector<double>(n_particle, 0));
    vector< vector<double> > x_resampled(T+1, vector<double>(n_particle, 0));

    normal_distribution<> dist1(0.0, 1.0);
    for(int i=0; i<n_particle; i++)
    {
      double initial_x = dist1(engine);
      x_resampled[0][i] = initial_x;
      x[0][i] = initial_x;
    }

    vector<double> y(T+1);
    y[0] = dist1(engine);

    vector< vector<double> > w(T, vector<double>(n_particle, 0));
    vector< vector<double> > w_normed(T, vector<double>(n_particle, 0));
    vector<double> l(T, 0);

    this->x = x;
    this->x_resampled = x_resampled;
    this->y = y;
    this->w = w;
    this->w_normed = w_normed;
    this->l = l;
    this->t_id = 0;
    this->tid = tid;
    this->width = n_particle / thread_num;
  }

  double norm_likelihood(double y, double x, double s2)
  {
    double result;
    result = pow(sqrt(2*M_PI*s2), -1)  * exp(pow(-(y-x), 2) / (2*s2));
    return result;
  }



  typedef struct data {
    ParticleFilter *ptr;
    int t;
    int tid;
  } DATA;

  void task(int t, int id)
  {
    Started[id] = true;
    int sid;
    struct timeval begin, end;

    while (true) {
      for (sid = 0; sid < Max_thread_num; sid++) {
	if (Started[sid] == false) break;
      }
      if (sid == Max_thread_num) break;
      else usleep(100);
    }
    if (ClockStarted == false) {
      ClockStarted = true;
      //StartTime = (double) clock();
      gettimeofday(&Begin, NULL);
    }

    gettimeofday(&begin, NULL);
    int width = this->width;
    int istart, iend;
    istart = id * width;
    iend = istart + width - 1;
    double v;
    Random rd;

    for(int i = istart; i < iend; i++) {
      // random のロックが邪魔していた？（std::normal_distribution）
      v = rd.rand_normal(0.0, sqrt(alpha_2*sigma_2));
      this->x[t+1][i] = x_resampled[t][i] + v;
      this->w[t][i] = norm_likelihood(y[t], x[t+1][i], sigma_2);
    }

    //double pe = (double) clock();
    //double pt = (pe - ps) / CLOCKS_PER_SEC;
    //printf("thread %d end" ,id);
    //printf(" / time: %.3f sec.\n", pt/this->thread_num);
    gettimeofday(&end, NULL);
    long diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
    // printf("id=%d: %ld\n", id, diff);

    pthread_exit(NULL);
  }

  static void* task_to_thread(void* t)
  {
    DATA* pair = static_cast<DATA*>(t);
    ParticleFilter* ptr = pair->ptr;
    int id = pair->t;
    int tid = pair->tid;
    ptr->task(id, tid);
    return NULL;
  }

  void parallel()
  {
    /* ---- multi ---- */
    for(int t=0; t<T; t++){
      for(int i=0; i<thread_num; i++)
      {
        DATA *data = (DATA*)malloc(sizeof(DATA));
        data->ptr = this;
        data->t =t;
        data->tid = i;
        pthread_create(&tid[i], NULL, ParticleFilter::task_to_thread, (void*)data);
      }
      for(int i=0; i<thread_num; i++)
      {
        pthread_join(tid[i], NULL);
      }

      double wt_sum = accumulate(w[t].begin(), w[t].end(),  0.0);
      for(int j=0; j<n_particle; j++)
      {
        this->w_normed[t][j] = w[t][j] / wt_sum;
      }
      this->l[t] = log(wt_sum);
    }
  }
};

int main(int argc, char *argv[])
{
  int n_particle = pow(10, atoi(argv[1]));
  int max_thread_num = atoi(argv[2]);
  Max_thread_num = max_thread_num;
  Started = (bool *)calloc(Max_thread_num, sizeof(bool));
  for (int i = 0; i < Max_thread_num; i++) Started[i] = false;

  //std::cout << "n_particle: " << n_particle << "  /  ";
  //std::cout << "max_thread_num: " << max_thread_num << std::endl;
  //double st = (double) clock();
  ParticleFilter pf(n_particle, sigma_2, alpha_2, max_thread_num);
  pf.parallel();

  gettimeofday(&End, NULL);
  long diff = (End.tv_sec - Begin.tv_sec) * 1000 * 1000 + (End.tv_usec - Begin.tv_usec);
  //double end = (double) clock();
  //double t = (end - StartTime) / CLOCKS_PER_SEC;
  printf("Exec time: %ld usec.\n", diff);
  return 0;
}
