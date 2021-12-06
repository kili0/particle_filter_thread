#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <math.h>
#include <time.h>
#include <pthread.h>

using namespace std;

int a = -2;
int b = -1;
double sigma_2 = pow(2, a);
double alpha_2 = pow(10, b);
int T = 1;
int time_len = 100;
int time_count = 0;
vector<double> cal_time(time_len);

class ParticleFilter
{
public:
  int n_particle;
  double sigma_2;
  double alpha_2;
  vector< vector<double> > x;
  vector< vector<double> > x_resampled;
  vector<double> y;
  vector< vector<double> > w;
  vector< vector<double> > w_normed;
  vector<double> l;
  int T;
  int time_count;
  vector<double> cal_time;

  void initialize(int n_particle, double sigma_2, double alpha_2)
  {
    n_particle = n_particle;
    sigma_2 = sigma_2;
    alpha_2 = alpha_2;

    vector< vector<double> > x(T+1, vector<double>(n_particle));
    vector< vector<double> > x_resampled(T+1, vector<double>(n_particle));

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    normal_distribution<> dist(0.0, 1.0);
    for(int i=0; i<n_particle; i++)
    {
      double initial_x = dist(engine);
      x_resampled[0][i] = initial_x;
      x[0][i] = initial_x;
    }
    vector<double> y(T);
    y[0] = dist(engine);

    vector< vector<double> > w(T, vector<double>(n_particle));
    vector< vector<double> > w_normed(T, vector<double>(n_particle));
    vector<double> l(T);
    vector<double> cal_time(time_len);
  }

  double norm_likelihood(double y, double x, double s2)
  {
    double result;
    result = pow(sqrt(2*M_PI*s2), -1)  * exp(pow(-(y-x), 2) / (2*s2));
    return result;
  }

  int task(int i, int t)
  {
    double v;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    normal_distribution<> dist(0.0, sqrt(alpha_2*sigma_2));
    v = dist(engine);

    x[t+1][i] = x_resampled[t][i] + v;
    w[t][i] = norm_likelihood(y[t], x[t+1][i], sigma_2);
    return 1;
  }

  void parallel()
  {
    clock_t start = clock();

    /* ---- ここがわからん ---- */

    clock_t end = clock();
    cal_time[time_count] = start - end;
    time_count++;
  }
};

int main(int argc, char *argv[])
{
  int n_particle = pow(10, atoi(argv[1]));
  int max_thread_num = atoi(argv[2]);

  std::cout << "n_particle: " << n_particle << std::endl;
  std::cout << "max_thread_num: " << max_thread_num << std::endl;

  ParticleFilter pf;
  pf.initialize(n_particle, sigma_2, alpha_2);
  pf.parallel();

  std::cout << "calculation time: "
            << accumulate(cal_time.begin(), cal_time.end(), 0.0)
            << std::endl;

}
