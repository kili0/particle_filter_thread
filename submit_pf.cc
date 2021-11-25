#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <math.h>
#include <time.h>
#include <thread>
#include <mutex>
#include <numeric>

#define SIZE_OF_ARRAY(array) (sizeof(array)/sizeof(array[0]))

using namespace std;

class ParticleFilter
{
public:
  int n_particle;
  double sigma_2;
  double alpha_2;
  int T;
  double x;
  double x_resampled;
  double w;
  double w_normed;
  double l;
  int time_count;
  double cal_time;
  mutex mtx;
};

void ParticleFilter::init(int n_particle, double sigma_2, double alpha_2)
{
  this->n_particle = n_particle;
  this->sigma_2 = sigma_2;
  this->alpha_2 = alpha_2;
}

double ParticleFilter::norm_likelihood(double y, double x, double s2)
{
  double result;
  result = pow(sqrt(2*M_PI*s2), -1)  * exp(pow(-(y-x), 2) / (2*s2));
  return result;
}

int ParticleFilter::task(int i, int t)
{
  double v;
  random_device seed_gen;
  default_random_engine engine(seed_gen());
  normal_distribution<> dist(0.0, sqrt(alpha_2*sigma_2));
  v = dist(engine);

  this->x[t+1, i] = this->x_resampled[t, i] + v;
  this->w[t, i] = norm_likelihood(this->y, this->x[t+1, i], this->sigma_2);
  return 1;
}

void ParticleFilter::prallel()
{
  clock_t start = clock();

  vector<thread> threads;
  /* ---- ここがわからん ---- */

  clock_t end = clock();
  this->cal_time[this->time_count] = start - end;
  this->time_count++
}


int main(char *argv[])
{
  int a = -2;
  int b = -1;

  int n_particle = 10 ** int(sys.argv[1]);
  double sigma_2 = 2**a;
  double alpha_2 = 10**b;

  int T = 1;
  array<array<double, n_particle>, T+1> x;
  // x.fill(0);
  array<array<double, n_particle>, T+1> x_resampled;

  random_device seed_gen;
  default_random_engine engine(seed_gen());
  normal_distribution<> dist(0.0, 1.0);
  for(int i=0; i<n_particle; i++)
  {
    double initial_x = dist(engine);
    x_resampled[0][i] = initial_x;
    x[0][i] = initial_x;
  }

  array<array<double, n_particle>, T> w;
  array<array<double, n_particle>, T> w_normed;
  array<double, T> l;

  int max_thread_num = int(argv[2]);
  width = int(n_particle / max_thread_num)

  /// for debug
  int time_len = 100
  array<double, time_len> cal_time;
  int time_count = 0

  cout << "n_particle: " << n_particle << endl;
  cout << "max_thread_num: " << max_thread_num << endl;

  ParticleFilter pf;
  pf.init(n_particle, sigma_2, alpha_2);
  pf.parallel();

  cout << "calculation time: "
       << accumulate(cal_time, cal_time + SIZE_OF_ARRAY(cal_time), 0.0) << endl;

}
