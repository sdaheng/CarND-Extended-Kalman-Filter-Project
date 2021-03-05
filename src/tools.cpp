#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    return rmse;
  }
  
  for (unsigned int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (px * py == 0) {
      return Hj;
  }
  // compute the Jacobian matrix
  float sumPxPy = px * px + py * py;
  float sqrtPxPy = sqrt(sumPxPy);
  
  if (fabs(sumPxPy) < 0.0001) {
    return Hj;
  }

  Hj << px / sqrtPxPy, py / sqrtPxPy, 0, 0,
        -(py / sumPxPy), px / sumPxPy, 0, 0,
        py * (vx * py - vy * px) / (sumPxPy * sqrtPxPy), px * (vy * px - vx * py) / (sumPxPy * sqrtPxPy), px / sqrtPxPy, py / sqrtPxPy;

  return Hj;
}
