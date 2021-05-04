#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
 // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<    1,   0,   0,   0,   0,
           0,   1,   0,   0,   0,
           0,   0,   1,   0,   0,
           0,   0,   0,   0.0225,   0,
           0,   0,   0,   0,   0.0225;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_aug_ = 7;
  lambda_ = 3-n_x_;
  n_z_lidar_ = 2;
  n_z_radar_ = 3;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; ++i) {  
    double weight = 0.5/(lambda_+n_aug_);
    weights_(i) = weight;
  }
  Tc_lidar_ =  MatrixXd(n_x_, n_z_lidar_);
  Tc_lidar_.fill(0.0);
  Tc_radar_ =  MatrixXd(n_x_, n_z_radar_);
  Tc_radar_.fill(0.0);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_){
      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
      float px = rho * std::cos(phi);
      float py = rho * std::sin(phi);
      float vx = rho_dot * std::cos(phi);
      float vy = rho_dot * std::sin(phi);
      float v  = sqrt(vx * vx + vy * vy); 

        x_ << px,py,v,0.0,0.0;
		time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
    } 
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      float x = meas_package.raw_measurements_[0];
      float y = meas_package.raw_measurements_[1];
      x_ << x,y,0.0,0.0,0.0;
      is_initialized_ = true;
	  time_us_ = meas_package.timestamp_;
    }
  } else {
    double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
    time_us_ = meas_package.timestamp_;
    // ------------------prediction of measurement----------------------------------
    Prediction(delta_t);
    // -------------------perform update of state-----------------------------------
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        UpdateRadar(meas_package);
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
        UpdateLidar(meas_package);
    }  
  }  
}

void UKF::Prediction(double delta_t) {
 VectorXd x_aug_ = VectorXd(n_aug_);
  // create augmented mean state
  x_aug_.head(5) = x_  ;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  // create augmented state covariance
  MatrixXd P_aug_ = MatrixXd(7, 7);
  // create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

   // create square root matrix
  MatrixXd L_ = P_aug_.llt().matrixL();
  // create sigma point matrix
  MatrixXd Xsig_aug_ =  MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);;

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L_.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L_.col(i);
  }

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);
	

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
		x_ +=  weights_(i) * Xsig_pred_.col(i);
	}
	
	// predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		P_ += weights_(i) * x_diff * x_diff.transpose() ;
	}
}


void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  MatrixXd Zsig = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  VectorXd z_pred = VectorXd(n_z_lidar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    
    Zsig.col(i) << px,
                   py;
    
    //calculating mean predicted measurement
    z_pred = z_pred + (weights_(i) * Zsig.col(i));
  }
	
	// measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_lidar_,n_z_lidar_);
    S.fill(0.0);
    /// Predicted Covariance
    for (int i = 0; i < 2 * n_aug_ + 1; i++){
      VectorXd zDiff = Zsig.col(i) - z_pred;
      S +=  weights_(i) * zDiff * zDiff.transpose();
    }
    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z_lidar_, n_z_lidar_);
    R << pow(std_laspx_ , 2), 0,
        0, pow(std_laspy_ , 2);
    S = S + R;	
	Tc_lidar_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    if (x_diff(3) > M_PI) {
        x_diff(3) -= 2. * M_PI;
    } else if (x_diff(3) < -M_PI){
      x_diff(3) += 2. * M_PI;
    }  
    Tc_lidar_ += weights_(i) * x_diff * z_diff.transpose();
  }
	//LIDAR measurement
  VectorXd z = meas_package.raw_measurements_;
  // Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z_lidar_);
  K.fill(0.0);
  K = Tc_lidar_ * S.inverse();
  // residual
  VectorXd z_diff = z - z_pred;
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  

    //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_,n_z_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_radar_,n_z_radar_);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;
  Tc_radar_.fill(0.0);
// calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc_radar_ += weights_(i) * x_diff * z_diff.transpose();
  }
  //RADAR measurement
  VectorXd z = meas_package.raw_measurements_;
  // Kalman gain K;
  MatrixXd K = Tc_radar_ * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}
