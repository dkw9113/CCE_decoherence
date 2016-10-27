#include<iostream>
#include<cmath>
#include <cstdio>
#include<complex>
#include<time.h>
#include<stdlib.h>
#include<fstream>
#include<armadillo>
#define EIGEN_USE_LAPACKE_STRICT
#include "Dense"
#include <string.h>
#include <math.h>

#define hbar 0.66//ueV*ns
#define mu0 4*M_PI*1E-05*1.602
#define g_75As 0.95563
#define g_71Ga 1.7026143
#define g_69Ga 1.33956
#define g13c 1.404239487
#define gs 2.00
#define m_e 9.10938356e-31
#define m_p 1.672621898e-27
#define mu_e 58 //ueV/T
#define mu_n mu_e*m_e/m_p
#define a0 0.35668

//TAKEN FROM nuclear_bath_QD_NNA_isotope_separate.py in spin_32 subfolde
using namespace std;
using namespace Eigen;
using namespace arma;
double omega(double magn_field){
	double w;
	w=-0.5*g13c*mu_n*magn_field;
	return w;
}
#if 0
void rotate_frame(const Matrix3d& Rij){
    Rij<<-1/sqrt(6)<<-1/sqrt(6)<<2/sqrt(6)<<1/sqrt(2)<<-1/sqrt(2)<<0<<1/sqrt(3)<<1/sqrt(3)<<1/sqrt(3);
}
#endif
double Axz(const Matrix3d& Rij, const Vector3d& r_iv1, const Vector3d& NV_loc, const Vector3d& NV_orient){
	Vector3d r_iv(0,0,0);
	r_iv=r_iv1-NV_loc;
	r_iv=(Rij*r_iv);
	double A1,b_sj;
	b_sj=-2*g13c*2.93501e-5;
	A1=b_sj*(-3*r_iv[0]*r_iv[2]/(pow(r_iv.norm(),2)))/(pow(r_iv.norm(),3));
	return A1;
}
double Ayz(const Matrix3d& Rij, const Vector3d& r_iv1, const Vector3d& NV_loc, const Vector3d& NV_orient){
	Vector3d r_iv(0,0,0);
	r_iv=r_iv1-NV_loc;
	r_iv=(Rij*r_iv);
	double A1,b_sj;
	b_sj=-2*g13c*2.93501e-5;
	A1=b_sj*(-3*r_iv[1]*r_iv[2]/(pow(r_iv.norm(),2)))/(pow(r_iv.norm(),3));
	return A1;
}
double Azz(const Matrix3d& Rij, const Vector3d& r_iv1, const Vector3d& NV_loc, const Vector3d& NV_orient){
	Vector3d r_iv(0,0,0);
	r_iv=r_iv1-NV_loc;
	r_iv=(Rij*r_iv);
	double A1,b_sj;
	b_sj=-2*g13c*2.93501e-5;
	A1=b_sj*(1-(3*r_iv[2]*r_iv[2]/(pow(r_iv.norm(),2))))/(pow(r_iv.norm(),3));
	return A1;
}
double A(const Vector3d& r_iv1, const Vector3d& NV_loc, const Vector3d& NV_orient){
	Vector3d r_iv(0,0,0);
	//double b_sj, z, R, nu_0,b,psi_z,psi_xy,psi2,g_i;
	double A1,b_sj;//,A11,A12;
	//A11=0;
	//A12=0;
	r_iv=(r_iv1-NV_loc);
	b_sj=-2*g13c*2.93501e-5; 

/*	for (int i=0;i<3;i++){
		printf("r_iv is [%f] .\n\n", r_iv[i]);
	}*/

	A1=(b_sj*(1-(3*(pow(((r_iv.dot(NV_orient))/(r_iv.norm()*(NV_orient.norm()))),2)))))/(pow(r_iv.norm(),3));
	return A1;
}
double b(const Vector3d& r_i, const Vector3d& r_j,const Vector3d& NV_orient){
	//Vector3d r_ij(0,0,0);
	double b_jj,b1;
	//r_ij=(r_i-r_j); 
	b_jj=1.5984568e-8*g13c*g13c;
	b1=(b_jj*(1-3*((pow((((r_i-r_j).dot(NV_orient))/((r_i-r_j).norm()*(NV_orient.norm()))),2))))/(pow((r_i-r_j).norm(),3)));
	return b1;
}

void ham1(cx_mat& ham, double sign, double Axz, double Ayz, double Azz, double omega){
 ham.eye();
 complex<double> j(0,1);
 //vec Iz = {1.5, 0.5, -0.5, -1.5};
 ham(0,1)=sign*0.25*(Axz-j*Ayz);
 ham(1,0)=sign*0.25*(Axz+j*Ayz);
 ham(0,0) = omega+sign*0.25*Azz;
 ham(1,1) = -omega-sign*0.25*Azz;
}

cx_vec cce1(cx_mat& hamplus1,cx_mat& hamminus1, cx_mat& Uplus1,  cx_mat& Uminus1, VectorXd& t, double Axz,double Ayz, double Azz, double omega){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();

	ham1(hamplus1, 1., Axz,Ayz,Azz, omega);
	//cout<<hamplus1<<endl;
	ham1(hamminus1, -1., Axz, Ayz, Azz, omega);
	for(int i=0; i<t.size(); i++){
	Uplus1 = expmat(-j*(t[i]/2)/hbar*hamplus1);

	//Uplus21 = expmat(j*(t/2)*hamplus1);

	Uminus1 = expmat(-j*(t[i]/2)/hbar*hamminus1);

	//Uminus21 = expmat(j*(t/2)*hamminus1);

	W[i]=(1./2.)*trace(Uplus1.t()*Uminus1.t()*Uplus1*Uminus1);

}
	return W;
}

void ham2(cx_mat& ham, double sign, vector<double> Axz, vector<double> Ayz,vector<double> Azz, vector<double> omega, double b12 ){
	complex<double> j(0,1);
	ham.eye();
	ham(0,2)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(1,3)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(2,0)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(3,1)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(0,1)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(1,0)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(2,3)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(3,2)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(0,0)=omega[0]+omega[1]+sign*0.25*(Azz[0]+Azz[1])-b12;
	ham(1,1)=omega[0]-omega[1]+sign*0.25*(Azz[0]-Azz[1])+b12;
	ham(2,2)=-omega[0]+omega[1]+sign*0.25*(-Azz[0]+Azz[1])+b12;
	ham(3,3)=-omega[0]-omega[1]+sign*0.25*(-Azz[0]-Azz[1])-b12;
	ham(1,2)=b12;
	ham(2,1)=b12;
}

cx_vec cce2(cx_mat& hamplus2,cx_mat& hamminus2, cx_mat& Uplus2,  cx_mat& Uminus2, VectorXd& t, vector<double> Axz, vector<double> Ayz,vector<double> Azz, vector<double> omega,double b12){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();
	
	ham2(hamplus2, 1., Axz, Ayz, Azz, omega,b12);
	ham2(hamminus2, -1., Axz, Ayz, Azz, omega,b12);
	//cout<<"hamminus: \n"<<hamminus2<<"\n";
	//cout<<"t.size(): "<<t.size()<<"\n";
	for(int i=0; i<t.size(); i++){
	//cout<<"loop start i:"<<i<<"\n";
	Uplus2 = expmat(-j*(t[i]/2)/hbar*hamplus2);
	//cout<<"Uplus2 created\n"<<Uplus2<<"\n";
	//Uplus21 = expmat(j*(t/2)*hamplus1);

	Uminus2 = expmat(-j*(t[i]/2)/hbar*hamminus2);
	//cout<<"Uminus2 created\n"<<Uminus2<<"\n";
	//Uminus21 = expmat(j*(t/2)*hamminus1);

	W[i]=(1./4.)*trace(Uplus2.t()*Uminus2.t()*Uplus2*Uminus2);
	//cout<<"Coherence calculated W[i]: "<<W[i]<<"\n";
	//cout<<"i: "<<i<<"\n";
}
	
	return W;
}
vec cce2_anal(VectorXd& t, vector<double>A1, vector<double> omega, double b12){
	vec Wan(t.size());
	Wan.ones();
	double C=b12;
	double A=(A1[0]-A1[1])/2;
	double w=(omega[0]-omega[1])/2;
	for(int i=0;i<t.size();i++){
		Wan[i]=1-16*C*C*A*A*((sin(t[i]/4*sqrt(4*C*C+(A-w)*(A-w))/hbar))*(sin(t[i]/4*sqrt(4*C*C+(A-w)*(A-w))/hbar))/(4*C*C+(A-w)*(A-w)))*((sin(t[i]/4*sqrt(4*C*C+(A+w)*(A+w))/hbar))*(sin(t[i]/4*sqrt(4*C*C+(A+w)*(A+w))/hbar))/(4*C*C+(A+w)*(A+w)));
		//cout<<"Wan[i]: "<<Wan[i]<<endl;
	}
	return Wan;
}

void ham3(cx_mat& ham, double sign, vector<double> Axz, vector<double> Ayz,vector<double> Azz, vector<double> omega, vector<double> bij ){
	/* VECTOR bij: {b12,b23,b13}*/
	complex<double> j(0,1);
	ham.eye();
	ham(0,0)=omega[0]+omega[1]+omega[2]+sign*0.25*(Azz[0]+Azz[1]+Azz[2])-bij[0]-bij[1]-bij[2];
	ham(1,1)=omega[0]+omega[1]-omega[2]+sign*0.25*(Azz[0]+Azz[1]-Azz[2])-bij[0]+bij[1]+bij[2];
	ham(2,2)=omega[0]-omega[1]+omega[2]+sign*0.25*(Azz[0]-Azz[1]+Azz[2])+bij[0]+bij[1]-bij[2];
	ham(3,3)=omega[0]-omega[1]-omega[2]+sign*0.25*(Azz[0]-Azz[1]-Azz[2])+bij[0]-bij[1]+bij[2];
	ham(4,4)=-omega[0]+omega[1]+omega[2]+sign*0.25*(-Azz[0]+Azz[1]+Azz[2])+bij[0]-bij[1]+bij[2];
	ham(5,5)=-omega[0]+omega[1]-omega[2]+sign*0.25*(-Azz[0]+Azz[1]-Azz[2])+bij[0]+bij[1]-bij[2];
	ham(6,6)=-omega[0]-omega[1]+omega[2]+sign*0.25*(-Azz[0]-Azz[1]+Azz[2])-bij[0]+bij[1]+bij[2];
	ham(7,7)=-omega[0]-omega[1]-omega[2]+sign*0.25*(-Azz[0]-Azz[1]-Azz[2])-bij[0]-bij[1]-bij[2];
	ham(2,4)=bij[0];
	ham(3,5)=bij[0];
	ham(4,2)=bij[0];
	ham(5,3)=bij[0];
	ham(1,2)=bij[1];
	ham(2,1)=bij[1];
	ham(5,6)=bij[1];
	ham(6,5)=bij[1];
	ham(1,4)=bij[2];
	ham(3,6)=bij[2];
	ham(4,1)=bij[2];
	ham(6,3)=bij[2];
	ham(0,4)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(1,5)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(2,6)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(3,7)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(4,0)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(5,1)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(6,2)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(7,3)=sign*0.25*(Axz[0]+j*Ayz[0]);

	ham(0,2)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(1,3)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(2,0)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(3,1)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(4,6)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(5,7)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(6,4)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(7,5)=sign*0.25*(Axz[1]+j*Ayz[1]);

	ham(0,1)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(1,0)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(2,3)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(3,2)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(4,5)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(5,4)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(6,7)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(7,6)=sign*0.25*(Axz[2]+j*Ayz[2]);
}
cx_vec cce3(cx_mat& hamplus3,cx_mat& hamminus3, cx_mat& Uplus3,  cx_mat& Uminus3, VectorXd& t, vector<double> Axz, vector<double> Ayz,vector<double> Azz, vector<double> omega, vector<double> bij){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();
	
	ham3(hamplus3, 1., Axz, Ayz, Azz, omega,bij);
	ham3(hamminus3, -1., Axz, Ayz, Azz, omega,bij);
	//cout<<"hamminus: \n"<<hamminus3<<"\n";
	//cout<<"t.size(): "<<t.size()<<"\n";
	for(int i=0; i<t.size(); i++){
	//cout<<"loop start i:"<<i<<"\n";
	Uplus3 = expmat(-j*(t[i]/2)/hbar*hamplus3);
	//cout<<"Uplus3 created\n"<<Uplus3<<"\n";
	//Uplus21 = expmat(j*(t/2)*hamplus1);

	Uminus3 = expmat(-j*(t[i]/2)/hbar*hamminus3);
	//cout<<"Uminus2 created\n"<<Uminus2<<"\n";
	//Uminus21 = expmat(j*(t/2)*hamminus1);

	W[i]=(1./8.)*trace(Uplus3.t()*Uminus3.t()*Uplus3*Uminus3);
	//cout<<"Coherence calculated W[i]: "<<W[i]<<"\n";
	//cout<<"i: "<<i<<"\n";
}
	
	return W;
}

void ham4(cx_mat& ham, double sign, vector<double> Axz, vector<double> Ayz, vector<double> Azz, vector<double> omega, vector<double> bij ){
	/* VECTOR bij: {b12,b23,b13}*/
	complex<double> j(0,1);
	ham.eye();
	ham(0,0)=omega[0]+omega[1]+omega[2]+omega[3]+sign*0.25*(Azz[0]+Azz[1]+Azz[2]+Azz[3])-bij[0]-bij[1]-bij[2]-bij[3]-bij[4]-bij[5];
	ham(1,1)=omega[0]+omega[1]+omega[2]-omega[3]+sign*0.25*(Azz[0]+Azz[1]+Azz[2]-Azz[3])-bij[0]-bij[1]-bij[2]+bij[3]+bij[4]+bij[5];
	ham(2,2)=omega[0]+omega[1]-omega[2]+omega[3]+sign*0.25*(Azz[0]+Azz[1]-Azz[2]+Azz[3])-bij[0]+bij[1]+bij[2]-bij[3]-bij[4]+bij[5];
	ham(3,3)=omega[0]+omega[1]-omega[2]-omega[3]+sign*0.25*(Azz[0]+Azz[1]-Azz[2]-Azz[3])-bij[0]+bij[1]+bij[2]+bij[3]+bij[4]-bij[5];
	ham(4,4)=omega[0]-omega[1]+omega[2]+omega[3]+sign*0.25*(Azz[0]-Azz[1]+Azz[2]+Azz[3])+bij[0]+bij[1]-bij[2]-bij[3]+bij[4]-bij[5];
	ham(5,5)=omega[0]-omega[1]+omega[2]-omega[3]+sign*0.25*(Azz[0]-Azz[1]+Azz[2]-Azz[3])+bij[0]+bij[1]-bij[2]+bij[3]-bij[4]+bij[5];
	ham(6,6)=omega[0]-omega[1]-omega[2]+omega[3]+sign*0.25*(Azz[0]-Azz[1]-Azz[2]+Azz[3])+bij[0]-bij[1]+bij[2]-bij[3]+bij[4]+bij[5];
	ham(7,7)=omega[0]-omega[1]-omega[2]-omega[3]+sign*0.25*(Azz[0]-Azz[1]-Azz[2]-Azz[3])+bij[0]-bij[1]+bij[2]+bij[3]-bij[4]-bij[5];
	ham(8,8)=-omega[0]+omega[1]+omega[2]+omega[3]+sign*0.25*(-Azz[0]+Azz[1]+Azz[2]+Azz[3])+bij[0]-bij[1]+bij[2]+bij[3]-bij[4]-bij[5];
	ham(9,9)=-omega[0]+omega[1]+omega[2]-omega[3]+sign*0.25*(-Azz[0]+Azz[1]+Azz[2]-Azz[3])+bij[0]-bij[1]+bij[2]-bij[3]+bij[4]+bij[5];
	ham(10,10)=-omega[0]+omega[1]-omega[2]+omega[3]+sign*0.25*(-Azz[0]+Azz[1]-Azz[2]+Azz[3])+bij[0]+bij[1]-bij[2]+bij[3]-bij[4]+bij[5];
	ham(11,11)=-omega[0]+omega[1]-omega[2]-omega[3]+sign*0.25*(-Azz[0]+Azz[1]-Azz[2]-Azz[3])+bij[0]+bij[1]-bij[2]-bij[3]+bij[4]-bij[5];
	ham(12,12)=-omega[0]-omega[1]+omega[2]+omega[3]+sign*0.25*(-Azz[0]-Azz[1]+Azz[2]+Azz[3])-bij[0]+bij[1]+bij[2]+bij[3]+bij[4]-bij[5];
	ham(13,13)=-omega[0]-omega[1]+omega[2]-omega[3]+sign*0.25*(-Azz[0]-Azz[1]+Azz[2]-Azz[3])-bij[0]+bij[1]+bij[2]-bij[3]-bij[4]+bij[5];
	ham(14,14)=-omega[0]-omega[1]-omega[2]+omega[3]+sign*0.25*(-Azz[0]-Azz[1]-Azz[2]+Azz[3])-bij[0]-bij[1]-bij[2]+bij[3]+bij[4]+bij[5];
	ham(15,15)=-omega[0]-omega[1]-omega[2]-omega[3]+sign*0.25*(-Azz[0]-Azz[1]-Azz[2]-Azz[3])-bij[0]-bij[1]-bij[2]-bij[3]-bij[4]-bij[5];
	ham(4,8)=bij[0];
	ham(5,9)=bij[0];
	ham(6,10)=bij[0];
	ham(7,11)=bij[0];
	ham(8,4)=bij[0];
	ham(9,5)=bij[0];
	ham(10,6)=bij[0];
	ham(11,7)=bij[0];
	ham(2,4)=bij[1];
	ham(4,2)=bij[1];
	ham(3,5)=bij[1];
	ham(5,3)=bij[1];
	ham(10,12)=bij[1];
	ham(11,13)=bij[1];
	ham(12,10)=bij[1];
	ham(13,11)=bij[1];
	ham(2,8)=bij[2];
	ham(3,9)=bij[2];
	ham(6,12)=bij[2];
	ham(7,13)=bij[2];
	ham(8,2)=bij[2];
	ham(9,3)=bij[2];
	ham(12,6)=bij[2];
	ham(13,7)=bij[2];
	ham(1,8)=bij[3];
	ham(3,10)=bij[3];
	ham(5,12)=bij[3];
	ham(7,14)=bij[3];
	ham(8,1)=bij[3];
	ham(10,3)=bij[3];
	ham(12,5)=bij[3];
	ham(14,7)=bij[3];
	ham(1,4)=bij[4];
	ham(3,6)=bij[4];
	ham(4,1)=bij[4];
	ham(6,3)=bij[4];
	ham(9,12)=bij[4];
	ham(11,14)=bij[4];
	ham(12,9)=bij[4];
	ham(14,11)=bij[4];
	ham(1,2)=bij[5];
	ham(2,1)=bij[5];
	ham(5,6)=bij[5];
	ham(6,5)=bij[5];
	ham(9,10)=bij[5];
	ham(10,9)=bij[5];
	ham(13,14)=bij[5];
	ham(14,13)=bij[5];

	ham(0,8)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(1,9)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(2,10)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(3,11)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(4,12)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(5,13)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(6,14)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(7,15)=sign*0.25*(Axz[0]-j*Ayz[0]);
	ham(8,0)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(9,1)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(10,2)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(11,3)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(12,4)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(13,5)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(14,6)=sign*0.25*(Axz[0]+j*Ayz[0]);
	ham(15,7)=sign*0.25*(Axz[0]+j*Ayz[0]);

	ham(0,4)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(1,5)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(2,6)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(3,7)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(4,0)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(5,1)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(6,2)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(7,3)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(8,12)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(9,13)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(10,14)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(11,15)=sign*0.25*(Axz[1]-j*Ayz[1]);
	ham(12,8)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(13,9)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(14,10)=sign*0.25*(Axz[1]+j*Ayz[1]);
	ham(15,11)=sign*0.25*(Axz[1]+j*Ayz[1]);

	ham(0,2)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(1,3)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(2,0)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(3,1)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(4,6)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(5,7)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(6,4)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(7,5)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(8,10)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(9,11)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(10,8)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(11,9)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(12,14)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(13,15)=sign*0.25*(Axz[2]-j*Ayz[2]);
	ham(14,12)=sign*0.25*(Axz[2]+j*Ayz[2]);
	ham(15,13)=sign*0.25*(Axz[2]+j*Ayz[2]);

	ham(0,1)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(1,0)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(2,3)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(3,2)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(4,5)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(5,4)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(6,7)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(7,6)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(8,9)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(9,8)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(10,11)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(11,10)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(12,13)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(13,12)=sign*0.25*(Axz[3]+j*Ayz[3]);
	ham(14,15)=sign*0.25*(Axz[3]-j*Ayz[3]);
	ham(15,14)=sign*0.25*(Axz[3]+j*Ayz[3]);
}
cx_vec cce4(cx_mat& hamplus4,cx_mat& hamminus4, cx_mat& Uplus4,  cx_mat& Uminus4, VectorXd& t, vector<double> Axz, vector<double> Ayz, vector<double> Azz, vector<double> omega, vector<double> bij){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();
	
	ham4(hamplus4, 1., Axz, Ayz, Azz, omega,bij);
	ham4(hamminus4, -1., Axz, Ayz, Azz, omega,bij);
	//cout<<"hamminus: \n"<<hamminus2<<"\n";
	//cout<<"t.size(): "<<t.size()<<"\n";
	for(int i=0; i<t.size(); i++){
	//cout<<"loop start i:"<<i<<"\n";
	Uplus4 = expmat(-j*(t[i]/2)/hbar*hamplus4);
	//cout<<"Uplus2 created\n"<<Uplus2<<"\n";
	//Uplus21 = expmat(j*(t/2)*hamplus1);

	Uminus4 = expmat(-j*(t[i]/2)/hbar*hamminus4);
	//cout<<"Uminus2 created\n"<<Uminus2<<"\n";
	//Uminus21 = expmat(j*(t/2)*hamminus1);

	W[i]=(1./16.)*trace(Uplus4.t()*Uminus4.t()*Uplus4*Uminus4);
	//cout<<"Coherence calculated W[i]: "<<W[i]<<"\n";
	//cout<<"i: "<<i<<"\n";
}
	
	return W;
}

const int SAMPLES_DIM = 3;
template <typename Der>
void read_file_to_Eigen(Eigen::MatrixBase<Der> &data,std::string filename, int rows, int cols){
	
	std::fstream file;
	file.open(filename.c_str(), std::ios::in);
	if(!file.is_open()){}
	
	data.resize(rows,cols);
	double item;
	for(size_t i =0;i<rows;++i)
	{
		for(size_t j =0;j<cols;++j){
			file>>item;
			data(i,j)=item;
		}
	}
}


int main(){
	int nc13=377;
	Eigen::Matrix<double,Dynamic,Dynamic> c13_loc(nc13,3);
	Eigen::Matrix<double,Dynamic,Dynamic> NV_loc(1,3);
	Eigen::Matrix<double,Dynamic,Dynamic> NV_orient(1,3);
	Eigen::Matrix<double,Dynamic,Dynamic> Wan_py(100,2);
	read_file_to_Eigen(c13_loc,"c13_loc.dat", nc13, 3);
	read_file_to_Eigen(NV_loc,"NV_loc.dat", 1, 3);
	read_file_to_Eigen(NV_orient,"NV_orient.dat", 1, 3);
	read_file_to_Eigen(Wan_py,"Wan_result_PY.dat",100,2);
	//cout<<"NR OF SPINS IN THE SYSTEM: "<<c13_loc.rows()<<'\n';
	#if 0
	for(int i=0;i<c13_loc.rows();i++){
		cout<<c13_loc(i,0)<<"\t"<<c13_loc(i,1)<<"\t"<<c13_loc(i,2)<<"\n";
	}
	#endif
	Matrix3d Rij;
	//rotate_frame(Rij);
	Rij<<-1/sqrt(6),-1/sqrt(6),2/sqrt(6),1/sqrt(2),-1/sqrt(2),0,1/sqrt(3),1/sqrt(3),1/sqrt(3);
	int i,j,k,l,N,cce_trunc;
	double R;
	i=0;
	j=0;
	k=0;
	l=0;
	double magn_field=3000*1e-4;
	R=0.8;
	N=c13_loc.rows();
	double tolerance=1e-4;
	//cout<<N<<"\n";
	cce_trunc=4;
	//mat A6(3,cce_trunc);
	vector<double> A1;
	vector<double> Axz1;
	vector<double> Ayz1;
	vector<double> Azz1;
	vector<double> omega1;
	vector<double> bij;
	cx_mat hamplus1(2,2);
	cx_mat hamminus1(2,2);
	cx_mat hamplus2(4,4);
	cx_mat hamminus2(4,4);
	cx_mat hamplus3(8,8);
	cx_mat hamminus3(8,8);
	cx_mat hamplus4(16,16);
	cx_mat hamminus4(16,16);

	cx_mat Uplus1(2,2);
	//mat Uplus2(2,2);
	cx_mat Uminus1(2,2);
	//mat Uminus21(2,2);
	cx_mat Uplus2(4,4);
	//mat Uplus22(4,4);
	cx_mat Uminus2(4,4);
	//mat Uminus22(4,4);
	cx_mat Uplus3(8,8);
	//mat Uplus22(4,4);
	cx_mat Uminus3(8,8);

	cx_mat Uplus4(16,16);
	cx_mat Uminus4(16,16);
	//VectorXd t=Wan_py.col(0);
	VectorXd t=VectorXd::LinSpaced(101,0,800000);
	//cout<<"ROSJA"<<"\n";
	cx_vec W1(t.size());
	cx_vec W11(t.size());
	cx_vec W21(t.size());
	cx_vec W31(t.size());
	cx_vec W41(t.size());

	cx_vec W2(t.size());
	cx_vec W12(t.size());
	cx_vec W22(t.size());
	cx_vec W32(t.size());
	cx_vec W42(t.size());
	cx_vec W52(t.size());
	cx_vec W62(t.size());

	cx_vec W3(t.size());
	cx_vec W13(t.size());
	cx_vec W23(t.size());
	cx_vec W33(t.size());
	cx_vec W43(t.size());
	cx_vec W3res(t.size());

	cx_vec W4(t.size());
	cx_vec W14(t.size());

	vec Wan(t.size());

	W1.ones();
	W2.ones();
	W3.ones();
	W4.ones();
	Wan.ones();

	//A6=zeros(3,cce_trunc);
	while(i<N){
	//A6.row(0)=A(location.row(i),...);
	//cout<<"ROSJA1"<<"\n";
	A1.push_back(A(c13_loc.row(i),NV_loc.row(0),NV_orient.row(0)));
	Axz1.push_back(Axz(Rij,c13_loc.row(i),NV_loc.row(0),NV_orient.row(0)));
	Ayz1.push_back(Ayz(Rij,c13_loc.row(i),NV_loc.row(0),NV_orient.row(0)));
	Azz1.push_back(Azz(Rij,c13_loc.row(i),NV_loc.row(0),NV_orient.row(0)));
	//cout<<Axz1[0]<<"\t"<<Ayz1[0]<<"\t"<<Azz1[0]<<endl;
	//cout<<A1[0]<<endl;
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif
	//cout<<"ZIEMNIAKI"<<"\n";
	W11=cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, Axz1[0],Ayz1[0],Azz1[0], omega1[0]);
	W1=W1%W11;
	
	//cout<<i<<"\n";
	j=i+1;
	while(j<N){if((c13_loc.row(j)-c13_loc.row(i)).norm()<R){
	//cout<<"i: "<<i<<" j: "<<j<<"\n";
	A1.push_back(A(c13_loc.row(j),NV_loc.row(0),NV_orient.row(0)));
	Axz1.push_back(Axz(Rij,c13_loc.row(j),NV_loc.row(0),NV_orient.row(0)));
	Ayz1.push_back(Ayz(Rij,c13_loc.row(j),NV_loc.row(0),NV_orient.row(0)));
	Azz1.push_back(Azz(Rij,c13_loc.row(j),NV_loc.row(0),NV_orient.row(0)));
	
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<"A: "<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif

	bij.push_back(b(c13_loc.row(i),c13_loc.row(j),NV_orient.row(0)));
	W12=cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, Axz1, Ayz1, Azz1, omega1, bij[0]);
	W21=cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, Axz1[1],Ayz1[1],Azz1[1], omega1[1]);
	//cout<<b12<<"\n";
	W12=(W12/(W11%W21));
	for(int i=0; i<W12.size();i++){
		if(abs(W12[i])-1>tolerance) W12[i]=1;
	}
	W2=W2%W12;
	//cout<<"ROSJA2"<<"\n";
	Wan=Wan%cce2_anal(t, A1, omega1, bij[0]);
	k=j+1;
	while(k<N){if((c13_loc.row(k)-c13_loc.row(i)).norm()<R){
	A1.push_back(A(c13_loc.row(k),NV_loc.row(0),NV_orient.row(0)));
	Axz1.push_back(Axz(Rij,c13_loc.row(k),NV_loc.row(0),NV_orient.row(0)));
	Ayz1.push_back(Ayz(Rij,c13_loc.row(k),NV_loc.row(0),NV_orient.row(0)));
	Azz1.push_back(Azz(Rij,c13_loc.row(k),NV_loc.row(0),NV_orient.row(0)));

	omega1.push_back(omega(magn_field));
	bij.push_back(b(c13_loc.row(j),c13_loc.row(k),NV_orient.row(0)));
	bij.push_back(b(c13_loc.row(i),c13_loc.row(k),NV_orient.row(0)));
	W31=cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, Axz1[2],Ayz1[2],Azz1[2], omega1[2]);
	W22=cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, {Axz1[1],Axz1[2]}, {Axz1[1],Ayz1[2]}, {Azz1[1],Azz1[2]}, {omega1[1],omega1[2]}, bij[1]);
	W32=cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, {Axz1[0], Axz1[2]}, {Ayz1[0], Ayz1[2]}, {Azz1[0], Azz1[2]}, {omega1[0], omega1[2]}, bij[2]);
	W13=cce3(hamplus3, hamminus3, Uplus3, Uminus3, t, Axz1, Ayz1, Azz1, omega1, bij);
	W13=(W13/(W12%(W22/(W21%W31))%(W32/(W11%W31))%W11%W21%W31));
	for(int i=0;i<W13.size();i++){
		if(abs(W13[i])-1>tolerance) W13[i]=1;
	}
	W3=W3%W13;
	//W3=W3%(W13);
	l=k+1;
	while(l<N){if((c13_loc.row(l)-c13_loc.row(i)).norm()<R){
	A1.push_back(A(c13_loc.row(l),NV_loc.row(0),NV_orient.row(0)));
	Axz1.push_back(Axz(Rij,c13_loc.row(l),NV_loc.row(0),NV_orient.row(0)));
	Ayz1.push_back(Ayz(Rij,c13_loc.row(l),NV_loc.row(0),NV_orient.row(0)));
	Azz1.push_back(Azz(Rij,c13_loc.row(l),NV_loc.row(0),NV_orient.row(0)));
	omega1.push_back(omega(magn_field));
	//currently bij contains: { b12, b23, b13, } now writing to {b12, b23, b13, b14, b24, b34}
	bij.push_back(b(c13_loc.row(i),c13_loc.row(l),NV_orient.row(0)));
	bij.push_back(b(c13_loc.row(j),c13_loc.row(l),NV_orient.row(0)));
	bij.push_back(b(c13_loc.row(k),c13_loc.row(l),NV_orient.row(0)));
	W41=cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, Axz1[3], Ayz1[3], Azz1[3], omega1[3]);
	W42=cce2(hamplus2,hamminus2,Uplus2,Uminus2,t, {Axz1[0],Axz1[3]}, {Ayz1[0],Ayz1[3]}, {Azz1[0],Azz1[3]}, {omega1[0],omega1[3]},bij[3]);
	W52=cce2(hamplus2,hamminus2,Uplus2,Uminus2,t, {Axz1[1],Axz1[3]}, {Ayz1[1],Ayz1[3]}, {Azz1[1],Azz1[3]}, {omega1[1],omega1[3]},bij[4]);
	W62=cce2(hamplus2,hamminus2,Uplus2,Uminus2,t, {Axz1[2],Axz1[3]}, {Ayz1[2],Ayz1[3]}, {Azz1[2],Azz1[3]}, {omega1[2],omega1[3]},bij[5]);
	W23=cce3(hamplus3,hamminus3,Uplus3,Uminus3,t, {Axz1[0],Axz1[1],Axz1[3]}, {Ayz1[0],Ayz1[1],Ayz1[3]}, {Azz1[0],Azz1[1],Azz1[3]}, {omega1[0],omega1[1],omega1[3]},{bij[0],bij[4],bij[3]});
	W33=cce3(hamplus3,hamminus3,Uplus3,Uminus3,t,{Axz1[0],Axz1[2],Axz1[3]},{Ayz1[0],Ayz1[2],Ayz1[3]},{Azz1[0],Azz1[2],Azz1[3]},{omega1[0],omega1[2],omega1[3]},{bij[2],bij[5],bij[3]});
	W43=cce3(hamplus3,hamminus3,Uplus3,Uminus3,t,{Axz1[1],Axz1[2],Axz1[3]},{Ayz1[1],Ayz1[2],Ayz1[3]},{Azz1[1],Azz1[2],Azz1[3]},{omega1[1],omega1[2],omega1[3]},{bij[1],bij[5],bij[4]});
	/*!!!!!!!!!!!!!!!HERE START CCE-4!!!!!!!!!!!!!!!!!*/
	//W13=(W13/((W12/(W11%W21))%(W22/(W21%W31))%(W32/(W11%W31))%W11%W21%W31));
	W23=(W23/(W12%(W42/(W11%W41))%(W52/(W21%W41))%W11%W21%W41));
	W33=(W33/((W32/(W11%W31))%(W42/(W11%W41))%(W62/(W31%W41))%W11%W31%W41));
	W43=(W43/((W22/(W21%W31))%(W52/(W21%W41))%(W62/(W31%W41))%W21%W31%W41));
	//W12=(W12/(W11%W21));
	W22=(W22/(W21%W31));
	W32=(W32/(W11%W31));
	W42=(W42/(W11%W41));
	W52=(W52/(W21%W41));
	W62=(W62/(W31%W41));
	
	for(int i=0;i<W43.size();i++){
		//if(abs(W13[i])-1>tolerance) W13[i]=1;
			
		if(abs(W23[i])-1>tolerance) W23[i]=1;
		if(abs(W33[i])-1>tolerance) W33[i]=1;
		if(abs(W43[i])-1>tolerance) W43[i]=1;
		//if(abs(W12[i])-1>tolerance) W12[i]=1;
		if(abs(W22[i])-1>tolerance) W22[i]=1;
		if(abs(W32[i])-1>tolerance) W32[i]=1;
		if(abs(W42[i])-1>tolerance) W42[i]=1;
		if(abs(W52[i])-1>tolerance) W52[i]=1;
		if(abs(W62[i])-1>tolerance) W62[i]=1;

	}
	W14=(cce4(hamplus4,hamminus4,Uplus4,Uminus4, t, Axz1, Ayz1, Azz1, omega1, bij)/(W13%W23%W33%W43%W12%W22%W32%W42%W52%W62%W11%W21%W31%W41));
	for(int i=0;i<W14.size();i++){
		if(abs(W14[i])-1>tolerance) W14[i]=1;
	}
	W4=W4%W14;
	//cout<<"CCE-4: len(A)\t"<<A1.size()<<"\t len(omega)\t"<<omega1.size()<<"\t len(bij)\t"<<bij.size()<<endl;
	bij.pop_back();
	bij.pop_back();
	bij.pop_back();
	A1.pop_back();
	Axz1.pop_back();
	Ayz1.pop_back();
	Azz1.pop_back();
	omega1.pop_back();
	}
	l++;	
	}
	A1.pop_back();
	Axz1.pop_back();
	Ayz1.pop_back();
	Azz1.pop_back();
	omega1.pop_back();
	bij.pop_back();
	bij.pop_back();
	}
	k++;
	}
	A1.pop_back();
	Axz1.pop_back();
	Ayz1.pop_back();
	Azz1.pop_back();

	omega1.pop_back();
	bij.pop_back();
	}
	j++;
	}
	A1.pop_back();
	Axz1.pop_back();
	Ayz1.pop_back();
	Azz1.pop_back();
	omega1.pop_back();
	i++;
	}

	W2=W1%W2;
	W3=W2%W3;
	W4=W3%W4;
	//cout<<"t: "<<"\t"<<"W1: "<<"\t"<<"W2: "<<"\t"<<"\t"<<"Wan: "<<"Wan_py: "<<"\n";
	
	for(int i=0;i<t.size();i++){
		//cout<<abs(W1[i])<<"\t"<<abs(W2[i])<<"\t"<<Wan[i]<<"\t"<<Wan_py(i,1)<<"\n";
		cout<<t[i]<<"\t"<<abs(W1[i])<<"\t"<<abs(W2[i])<<"\t"<<abs(W3[i])<<"\t"<<abs(W4[i])<<"\t"<<Wan[i]<<"\n";
		//cout<<t[i]<<"\t"<<Wan[i]<<"\t"<<Wan_py(i,1)<<"\n";
	}
	
	
}