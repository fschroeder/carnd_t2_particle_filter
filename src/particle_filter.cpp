/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 80;

	// "make some noise^^"
	normal_distribution<double> nd_x(x, std[0]);
	normal_distribution<double> nd_y(y, std[1]);
	normal_distribution<double> nd_t(theta, std[2]);
	
	// Create Particles
	for(int i = 0; i < num_particles; i++){
		Particle par;

		par.id = i;
		par.x = nd_x(random_gen);
		par.y = nd_y(random_gen);
		par.theta = nd_t(random_gen);
		par.weight = 1.0;

		particles.push_back(par);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Create gaussian distribution
	normal_distribution<double> nd_x(0.0, std_pos[0]);
	normal_distribution<double> nd_y(0.0, std_pos[1]);
	normal_distribution<double> nd_t(0.0, std_pos[2]);

	// Predict every particle
	for(int i = 0; i < particles.size(); i++){

		// No devision by zero - It's like sharing cookies with no friends...
		if(fabs(yaw_rate) < 0.0001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + delta_t * yaw_rate) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + delta_t * yaw_rate));
			particles[i].theta += delta_t * yaw_rate;
		}

		// The Beastie Boys are "make some noise" ^^
		particles[i].x += nd_x(random_gen);
		particles[i].y += nd_y(random_gen);
		particles[i].theta += nd_t(random_gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	// If there are no predictions nor oberservations -> nothing to do..
	if((predicted.size() > 0) && (observations.size() > 0)){

		// Iterate through all observations
		for(int i = 0; i < observations.size(); i++){

			// Init variables with the first value
			double min_distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
			int closest_prediction_id = predicted[0].id;

			// Iterate through every prediction and check for the shortest distance (remeber to start at index 1!!!!)
			for(int j = 1; j < predicted.size(); j++){
				double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
				if(current_distance < min_distance){
					closest_prediction_id = predicted[j].id;
					min_distance = current_distance;
				}
			}
			observations[i].id = closest_prediction_id;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Iterate through all particles...
	for(int i = 0; i < num_particles; i++){

		// Quite often used variables
		const double par_x = particles[i].x;
		const double par_y = particles[i].y;
		const double par_t = particles[i].theta;

		// Transform observations from vehicle coordiniations to map coordinates
		// ============================================================================
		std::vector<LandmarkObs> observations_t;

		for(int o = 0; o < observations.size(); o++ ){
			const double xm = par_x + cos(par_t) * observations[o].x - sin(par_t) * observations[o].y;
			const double ym = par_y + sin(par_t) * observations[o].x + cos(par_t) * observations[o].y;

			observations_t.push_back(LandmarkObs{observations[o].id, xm, ym});
		}

		// Create a list with all landmarks in sensor range
		// ============================================================================
		std::vector<LandmarkObs> predictions;

		for(int m = 0; m < map_landmarks.landmark_list.size(); m++){

			const int lm_id = map_landmarks.landmark_list[m].id_i;
			const float lm_x = map_landmarks.landmark_list[m].x_f;
			const float lm_y = map_landmarks.landmark_list[m].y_f;

			const double distance = dist(par_x, par_y, lm_x, lm_y);

			if(distance <= sensor_range){
				predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}

		// Associate (closes) prediction with observation
		// ============================================================================
		dataAssociation(predictions, observations_t);

		// Update particle's weight
		// ============================================================================
		particles[i].weight = 1.0;

		// Iterate through all transformed observations and recalculate weight
		for(int ot = 0; ot < observations_t.size(); ot++){

			double x_ = std::numeric_limits<double>::max();
			double y_ = std::numeric_limits<double>::max();

			// Find corresponding prediction (It's getting to a point were I don't like iterartions anymore... HashMaps - where are you? )
			for(int p = 0; p < predictions.size(); p++){
				if(predictions[p].id == observations_t[ot].id){
					x_ = predictions[p].x;
					y_ = predictions[p].y;
					break;
				}
			}

			// Just some copy paste, but easier for the following equation
			const double mu_x = observations_t[ot].x; 
			const double mu_y = observations_t[ot].y;
			const double sigma_x = std_landmark[0];
			const double sigma_y= std_landmark[1];

			// Calulate Multivariate-Gaussian probability density
			const double mgpd = (1/(2*M_PI*sigma_x*sigma_y)) * 
				exp( -( pow(x_ - mu_x,2)/(2*pow(sigma_x, 2)) + (pow(y_ - mu_y,2)/(2*pow(sigma_y, 2))) ) );

			particles[i].weight *= mgpd;

		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> particles_resampled;

	// Get max() of all weights
	std::vector<double> resample_weights;
	for(int i = 0; i < particles.size(); i++){
		resample_weights.push_back(particles[i].weight);
	}
	const double max_weight = *max_element(resample_weights.begin(), resample_weights.end());

	// Create random index
	uniform_int_distribution<int> uint_dist(0, particles.size()-1);
	auto index = uint_dist(random_gen);

	// Create beta value 
	uniform_real_distribution<double> uint_real_dist(0.0, (max_weight * 2.0));
	double beta = 0.0;

	// Implementation of resampling wheel
	for(int i = 0; i < particles.size(); i++){
		beta = uint_real_dist(random_gen);
		while(beta > resample_weights[index]){
			beta -= resample_weights[index];
			index = fmod((index + 1),particles.size());
		}
		particles_resampled.push_back(particles[index]);
	}

	particles = particles_resampled;


}

// Dead code !?
/*Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
} */

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
} 
