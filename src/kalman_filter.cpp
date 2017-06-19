#include "kalman_filter.h"
#include "tools.h"
#include <iostream>
#include <cmath> 

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  Update_(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd z_pred = ComputeRadarMeasurementFromState_();
  VectorXd y = NormalizeAngleInRadarMeasurement_(z - z_pred);
  Update_(y);
}

void KalmanFilter::Update_(const VectorXd& y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::ComputeRadarMeasurementFromState_() {
  VectorXd z(3);

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float c1 = px * px + py * py;

  if (fabs(px) < 0.001 || fabs(c1) < 0.0001) {
    std::cout << "ComputeRadarMeasurementFromState_() - Error - Division by Zero" << std::endl;
    return z;
  }

  z << sqrt(c1), atan2(py, px), (px * vx + py * vy) / sqrt(c1);

  return z;
}

VectorXd KalmanFilter::NormalizeAngleInRadarMeasurement_(const VectorXd& z) {
  VectorXd z_norm = z;

  // Normalize theta to -pi -> pi.
  float theta = z(1);
  theta = fmod(theta + M_PI, 2 * M_PI);
  if (theta < 0) {
    theta += 2 * M_PI;
  }
  theta -= M_PI;

  z_norm(1) = theta;
  return z_norm;
}
