#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <numeric>

#define a -2
#define b -1
#define T  1

using namespace std;

double sigma_2 = pow(2, a);
double alpha_2 = pow(10, b);
// int time_count = 0;

random_device seed_gen;
default_random_engine engine(seed_gen());

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

public:
  ParticleFilter(int np, double s2, double a2)
  {
    n_particle = np;
    sigma_2 = s2;
    alpha_2 = a2;

    // T = 1;
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
    vector<double> cal_time(T);

    this->x = x;
    this->x_resampled = x_resampled;
    this->y = y;
    this->w = w;
    this->w_normed = w_normed;
    this->l = l;
    // this->cal_time = cal_time;
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
    normal_distribution<> dist2(0.0, sqrt(alpha_2*sigma_2));
    v = dist2(engine);
    this->x[t+1][i] = x_resampled[t][i] + v;
    this->w[t][i] = norm_likelihood(y[t], x[t+1][i], sigma_2);
    return 1;
  }

  void parallel()
  {
    // clock_t start = clock();

    /* ---- single ---- */
    for(int t=0; t<T; t++)
    {
      for(int i=0; i<n_particle; i++)
      {
        task(i, t);
      }

      double wt_sum = accumulate(w[t].begin(), w[t].end(),  0.0);
      for(int j=0; j<n_particle; j++)
      {
        this->w_normed[t][j] = w[t][j] / wt_sum;
      }
      this->l[t] = log(wt_sum);
    }
    // clock_t end = clock();
    // cal_time[time_count] = end - start;
    // time_count++;
  }

/*
  double getCalTime()
  {
    double result;
    result = cal_time[0] / CLOCKS_PER_SEC;
    return result;
  }
*/

  void printVectorX()
  {
    std::cout << "x: ";
    for(int t=0; t<T+1; t++)
    {
      for(int i=0; i<n_particle; i++)
      {
        std::cout << x[t][i] << ", ";
      }
    }
    std::cout << std::endl;

    std::cout << "x_resampled: ";
    for(int t=0; t<T+1; t++)
    {
      for(int i=0; i<n_particle; i++)
      {
        std::cout << x_resampled[t][i] << ", ";
      }
    }
    std::cout << std::endl;
  }

  void printVectorW()
  {
    std::cout << "w: ";
    for(int t=0; t<T; t++)
    {
      for(int i=0; i<n_particle; i++)
      {
        std::cout << w[t][i] << ", ";
      }
    }
    std::cout << std::endl;

    std::cout << "w_normed: ";
    for(int t=0; t<T; t++)
    {
      for(int i=0; i<n_particle; i++)
      {
        std::cout << w_normed[t][i] << ", ";
      }
    }
    std::cout << std::endl;
  }
};

int main(int argc, char *argv[])
{
  int n_particle = pow(10, atoi(argv[1]));
  int max_thread_num = atoi(argv[2]);

  std::cout << "n_particle: " << n_particle << "  /  ";
  std::cout << "max_thread_num: " << max_thread_num << std::endl;

  ParticleFilter pf(n_particle, sigma_2, alpha_2);
  pf.parallel();

  // double result_time = pf.getCalTime();

  // pf.printVectorX();
  // pf.printVectorW();

  // std::cout << "calculation time: "
  //          << result_time
  //          << std::endl;

  return 0;
}
