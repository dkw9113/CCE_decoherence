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
#define A_75As hbar*6.53e1
#define A_71Ga hbar*6.99e1
#define A_69Ga hbar*5.47e1


//TAKEN FROM nuclear_bath_QD_NNA_isotope_separate.py in spin_32 subfolde
using namespace std;
using namespace Eigen;
using namespace arma;
double omega(double magn_field){
	double w;
	w=-1/2*g13c*mu_n*magn_field;
	return w;
}
double A(const Vector3d& r_iv1, const Vector3d& NV_loc, const Vector3d& B_orient, double l, double z0, double a0,string spec){
	Vector3d r_iv(0,0,0);
	double b_sj, z, R, nu_0,b,psi_z,psi_xy,psi2,g_i;
	double A1,A11,A12;
	A11=0;
	A12=0;
	r_iv=r_iv1-NV_loc;
	b_sj=2*2.93501e-5;
	z=z0/2-abs(r_iv[2]);
	R=pow(r_iv[0],2)+pow(r_iv[1],2)-pow(l,2);
	nu_0=pow(a0,3)*0.25;
	b=1/z0;
	//Fang-Howard below:
	//psi_z=sqrt(pow(b,3)/2)*r_iv[2]*exp(-b*r_iv[2]);
	psi_z=sqrt(2/z0)*cos(M_PI*r_iv[2]/z0);
	psi_xy=(1/(sqrt(M_PI*pow(l,2))))*exp(-(pow(r_iv[0],2)+pow(r_iv[1],2))/(2*pow(l,2)));
	psi2=nu_0*pow(psi_z,2)*pow(psi_xy,2);
	if(spec=="75As"){
		if(z>=0) A11=A_75As*psi2;
		if(R>=0) A12=-(b_sj*g_75As*(1-(3*(pow(((r_iv.dot(B_orient))/(r_iv.norm()*(B_orient.norm()))),2)))))/(pow(r_iv.norm(),3));
		A1=A11+A12;
		g_i=g_75As;
	}
	else if(spec=="69Ga"){
		if(z>=0) A11=A_69Ga*psi2;
		if(R>=0) A12=-(b_sj*g_69Ga*(1-(3*(pow(((r_iv.dot(B_orient))/(r_iv.norm()*(B_orient.norm()))),2)))))/(pow(r_iv.norm(),3));
		A1=A11+A12;
		g_i=g_69Ga;
	}
	else if(spec=="71Ga"){
		if(z>=0) A11=A_71Ga*psi2;
		if(R>=0) A12=-(b_sj*g_71Ga*(1-(3*(pow(((r_iv.dot(B_orient))/(r_iv.norm()*(B_orient.norm()))),2)))))/(pow(r_iv.norm(),3));
		A1=A11+A12;
		g_i=g_71Ga;
	}
/*	for (int i=0;i<3;i++){
		printf("r_iv is [%f] .\n\n", r_iv[i]);
	}*/
	return A1;
}
double b(const Vector3d& r_i, const Vector3d& r_j,const Vector3d& B_orient,string spec){
	//Vector3d r_ij(0,0,0);
	double b_jj,b1, g1;
	//r_ij=(r_i-r_j); 
	if(spec=="69Ga") g1=g_69Ga;
	if(spec=="71Ga") g1=g_71Ga;
	if(spec=="75As") g1=g_75As;
	b_jj=1.5984568e-8*g1*g1;
	b1=(b_jj*(1-3*((pow((((r_i-r_j).dot(B_orient))/((r_i-r_j).norm()*(B_orient.norm()))),2))))/(pow((r_i-r_j).norm(),3)));
	return b1;
}

void ham1(mat& ham, double sign, double A, double omega){
 ham.eye();
 //vec Iz = {1.5, 0.5, -0.5, -1.5};
 ham(0,0) = 3*omega+sign*0.75*A;
 ham(1,1) = omega+sign*0.25*A;
 ham(2,2) = -omega-sign*0.25*A;
 ham(3,3) = -3*omega-sign*0.75*A;
}

cx_vec cce1(mat& hamplus1,mat& hamminus1, cx_mat& Uplus1,  cx_mat& Uminus1, VectorXd& t, double A, double omega){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();

	ham1(hamplus1, 1., A, omega);

	ham1(hamminus1, -1., A, omega);
	for(int i=0; i<t.size(); i++){
	Uplus1 = expmat(-j*(t[i]/2)/hbar*hamplus1);

	//Uplus21 = expmat(j*(t/2)*hamplus1);

	Uminus1 = expmat(-j*(t[i]/2)/hbar*hamminus1);

	//Uminus21 = expmat(j*(t/2)*hamminus1);

	W[i]=(1./4.)*trace(Uplus1.t()*Uminus1.t()*Uplus1*Uminus1);

}
	return W;
}

void ham2(mat& ham, double sign, vector<double> A, vector<double> omega, double b12 ){
	ham.eye();
	ham(0,0)=3*omega[0]+3*omega[1]+sign*0.25*(3*A[0]+3*A[1])-9*b12;
	ham(1,1)=3*omega[0]+omega[1]+sign*0.25*(3*A[0]+A[1])-3*b12;
	ham(2,2)=3*omega[0]-omega[1]+sign*0.25*(3*A[0]-A[1])+3*b12;
	ham(3,3)=3*omega[0]-3*omega[1]+sign*0.25*(3*A[0]-3*A[1])+9*b12;
	ham(4,4)=omega[0]+3*omega[1]+sign*0.25*(A[0]+3*A[1])-3*b12;
	ham(5,5)=omega[0]+omega[1]+sign*0.25*(A[0]+A[1])-b12;
	ham(6,6)=omega[0]-omega[1]+sign*0.25*(A[0]-A[1])+b12;
	ham(7,7)=omega[0]-3*omega[1]+sign*0.25*(A[0]-3*A[1])+3*b12;
	ham(8,8)=-omega[0]+3*omega[1]+sign*0.25*(-A[0]+3*A[1])+3*b12;
	ham(9,9)=-omega[0]+omega[1]+sign*0.25*(-A[0]+A[1])+b12;
	ham(10,10)=-omega[0]-omega[1]+sign*0.25*(-A[0]-A[1])-b12;
	ham(11,11)=-omega[0]-3*omega[1]+sign*0.25*(-A[0]-3*A[1])-3*b12;
	ham(12,12)=-3*omega[0]+3*omega[1]+sign*0.25*(-3*A[0]+3*A[1])+9*b12;
	ham(13,13)=-3*omega[0]+omega[1]+sign*0.25*(-3*A[0]+A[1])+3*b12;
	ham(14,14)=-3*omega[0]-omega[1]+sign*0.25*(-3*A[0]-A[1])-3*b12;
	ham(15,15)=-3*omega[0]-3*omega[1]+sign*0.25*(-3*A[0]-3*A[1])-9*b12;
	ham(1,4)=3*b12;
	ham(2,5)=2*sqrt(3)*b12;
	ham(3,6)=3*b12;
	ham(4,1)=3*b12;
	ham(5,2)=2*sqrt(3)*b12;
	ham(5,8)=2*sqrt(3)*b12;
	ham(6,3)=3*b12;
	ham(6,9)=4*b12;
	ham(7,10)=2*sqrt(3)*b12;
	ham(8,5)=2*sqrt(3)*b12;
	ham(9,6)=4*b12;
	ham(9,12)=3*b12;
	ham(10,7)=2*sqrt(3)*b12;
	ham(10,13)=2*sqrt(3)*b12;
	ham(11,14)=3*b12;
	ham(12,9)=3*b12;
	ham(13,10)=2*sqrt(3)*b12;
	ham(14,11)=3*b12;
}

cx_vec cce2(mat& hamplus2,mat& hamminus2, cx_mat& Uplus2,  cx_mat& Uminus2, VectorXd& t, vector<double> A, vector<double> omega,double b12){
	complex<double> j(0,1);
	cx_vec W(t.size());
	W.ones();
	
	ham2(hamplus2, 1., A, omega,b12);
	ham2(hamminus2, -1., A, omega,b12);
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

	W[i]=(1./16.)*trace(Uplus2.t()*Uminus2.t()*Uplus2*Uminus2);
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
	int nga69=161569;
	int nga71=80538;
	int nas75=282384;
	Eigen::Matrix<double,Dynamic,Dynamic> ga69_loc(nga69,3);
	Eigen::Matrix<double,Dynamic,Dynamic> ga71_loc(nga71,3);
	Eigen::Matrix<double,Dynamic,Dynamic> as75_loc(nas75,3);
	//Eigen::Matrix<double,Dynamic,Dynamic> NV_orient(1,3);
	//Eigen::Matrix<double,Dynamic,Dynamic> Wan_py(100,2);
	read_file_to_Eigen(ga69_loc,"69Ga_dist_z0_10_l0_60.dat", nga69, 3);
	read_file_to_Eigen(ga71_loc,"71Ga_dist_z0_10_l0_60.dat", nga71, 3);
	read_file_to_Eigen(as75_loc,"75As_dist_z0_10_l0_60.dat", nas75, 3);

	double gamma_e, gamma_75As, gamma_71Ga, gamma_69Ga,  d_75As, d_71Ga, d_69Ga;
//double g_75As,g_71Ga,g_69Ga,gs,A_75As,A_71Ga,A_69Ga,mu_e,mu_n;

	gamma_e=-gs*mu_e/hbar;
	gamma_75As=g_75As*mu_n/hbar;
	gamma_71Ga=g_71Ga*mu_n/hbar;
	gamma_69Ga=g_69Ga*mu_n/hbar;

	d_75As=9.8e-4; //nm^-3
	d_71Ga=5.8e-4; //nm^-3
	d_69Ga=5.8e-4; //nm^-3
	//const complex<double> j(0,1);
/*From L. Cywinski et al., PRB 09, hf-mediated*/
	double l0,z0,a0;
	l0=60;
	z0=10;
	a0=0.565;
	Vector3d NV_loc;
	NV_loc<<l0,l0,z0/2;
	Vector3d B_orient;
	B_orient<<1,0,0;
	//read_file_to_Eigen(NV_orient,"NV_orient.dat", 1, 3);
	//read_file_to_Eigen(Wan_py,"Wan_result_PY.dat",100,2);
	//cout<<"NR OF SPINS IN THE SYSTEM: "<<c13_loc.rows()<<'\n';
	#if 0
	for(int i=0;i<c13_loc.rows();i++){
		cout<<c13_loc(i,0)<<"\t"<<c13_loc(i,1)<<"\t"<<c13_loc(i,2)<<"\n";
	}
	#endif
	int i,j,N,cce_trunc;
	double R;
	i=0;
	j=0;
	double b12;
	double magn_field=3000*1e-4;
	R=0.8;
	srand(time(0)); // use current time as seed for random generator
    //int random = rand()/RAND_MAX;
    double percent=0.01;
    long double random=0;
    //std::cout << "Random value on [0 " << RAND_MAX << "]: " 
	//cout<<N<<"\n";
	cce_trunc=2;
	//mat A6(3,cce_trunc);
	vector<double> A1;
	vector<double> omega1;
	mat hamplus1(4,4);
	mat hamminus1(4,4);
	mat hamplus2(16,16);
	mat hamminus2(16,16);
	cx_mat Uplus1(4,4);
	//mat Uplus2(2,2);
	cx_mat Uminus1(4,4);
	//mat Uminus21(2,2);
	cx_mat Uplus2(16,16);
	//mat Uplus22(4,4);
	cx_mat Uminus2(16,16);
	//mat Uminus22(4,4);
	//VectorXd t=Wan_py.col(0);
	VectorXd t=VectorXd::LinSpaced(11,0,60000);
	//cout<<"ROSJA"<<"\n";
	cx_vec W1(t.size());
	cx_vec W2(t.size());
	cx_vec W2_69Ga(t.size());
	cx_vec W2_71Ga(t.size());
	cx_vec W2_75As(t.size());
	//vec Wan(t.size());
	W1.ones();
	W2.ones();
	W2_69Ga.ones();
	W2_71Ga.ones();
	W2_75As.ones();
	//Wan.ones();
	int neighb=0;
	//A6=zeros(3,cce_trunc);
	N=ga69_loc.rows();
	while(i<N){
		neighb=0;
	//A6.row(0)=A(location.row(i),...);
	//cout<<"ROSJA1"<<"\n";
	A1.push_back(A(ga69_loc.row(i),NV_loc,B_orient,l0,z0,a0,"69Ga"));
	//cout<<A1[0]<<endl;
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif
	//cout<<"ZIEMNIAKI"<<"\n";
/*!!!!!!!! THESE LINES WITHIN precompiler if ARE IMPORTANT TO START CALCULATIONS FOR CCE-1 WITH PULSES DIFFERENT THAN HAHN ECHO !!!!!!!!!!!!!!!!!*/
		#if 0
		W1=W1%cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, A1[0], omega1[0]);
		#endif
	//cout<<i<<"\n";
	j=i+1;
	while(j<N){
		//random=((double) rand() / (RAND_MAX));
		//cout<<random<<endl;
		if((ga69_loc.row(j)-ga69_loc.row(i)).norm()<R){

			random=((double) rand() / (RAND_MAX));
			if(random<percent){
		//cout<<rand()/RAND_MAX<<endl;
	//cout<<"i: "<<i<<" j: "<<j<<"\n";
	A1.push_back(A(ga69_loc.row(j),NV_loc,B_orient,l0,z0,a0,"69Ga"));
	
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<"A: "<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif

	b12=b(ga69_loc.row(i),ga69_loc.row(j),B_orient,"69Ga");
	//cout<<b12<<"\n";
	W2_69Ga=W2_69Ga%cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, A1, omega1, b12);
	//cout<<"ROSJA2"<<"\n";
	//Wan=Wan%cce2_anal(t, A1, omega1, b12);

	A1.pop_back();

	omega1.pop_back();
	neighb++;
	}
	}
	j++;
	}
	A1.pop_back();
	omega1.pop_back();
	//cout<<"neighbours included: "<<neighb<<endl;
	i++;
	}
	for(int i=0;i<W2_69Ga.size();i++){
		W2_69Ga[i]=pow(abs(W2_69Ga[i]),1/percent);
	}
	
	W2=W2%W1%W2_69Ga;
	cout<<"69 Ga calculated."<<endl;
	N=ga71_loc.rows();
	i=0;
	j=0;
	while(i<N){
		neighb=0;
	//A6.row(0)=A(location.row(i),...);
	//cout<<"ROSJA1"<<"\n";
	A1.push_back(A(ga71_loc.row(i),NV_loc,B_orient,l0,z0,a0,"71Ga"));
	//cout<<A1[0]<<endl;
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif
	//cout<<"ZIEMNIAKI"<<"\n";
/*!!!!!!!! THESE LINES WITHIN precompiler if ARE IMPORTANT TO START CALCULATIONS FOR CCE-1 WITH PULSES DIFFERENT THAN HAHN ECHO !!!!!!!!!!!!!!!!!*/
		#if 0
		W1=W1%cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, A1[0], omega1[0]);
		#endif
	//cout<<i<<"\n";
	j=i+1;
	while(j<N){
		if((ga71_loc.row(j)-ga71_loc.row(i)).norm()<R){
			random=((double) rand() / (RAND_MAX));
			if(random<percent){
	//cout<<"i: "<<i<<" j: "<<j<<"\n";
	A1.push_back(A(ga71_loc.row(j),NV_loc,B_orient,l0,z0,a0,"71Ga"));
	
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<"A: "<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif

	b12=b(ga71_loc.row(i),ga71_loc.row(j),B_orient,"71Ga");
	//cout<<b12<<"\n";
	W2_71Ga=W2_71Ga%cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, A1, omega1, b12);
	//cout<<"ROSJA2"<<"\n";
	//Wan=Wan%cce2_anal(t, A1, omega1, b12);

	A1.pop_back();

	omega1.pop_back();
	neighb++;
	}
	}
	j++;
	}
	//cout<<"Neighbours included: "<<neighb<<endl;
	A1.pop_back();
	omega1.pop_back();
	i++;
	}
	for(int i=0;i<W2_71Ga.size();i++){
		W2_71Ga[i]=pow(abs(W2_71Ga[i]),1/percent);
	}
	//W2_71Ga=pow(abs(W2_71Ga),1/percent);
	W2=W2%W1%W2_71Ga;
	cout<<"71 Ga calculated."<<endl;
	N=as75_loc.rows();
	i=0;
	j=0;
	while(i<N){
		neighb=0;
	//A6.row(0)=A(location.row(i),...);
	//cout<<"ROSJA1"<<"\n";
	A1.push_back(A(as75_loc.row(i),NV_loc,B_orient,l0,z0,a0,"75As"));
	//cout<<A1[0]<<endl;
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif
	//cout<<"ZIEMNIAKI"<<"\n";
	/*!!!!!!!! THESE LINES WITHIN precompiler if ARE IMPORTANT TO START CALCULATIONS FOR CCE-1 WITH PULSES DIFFERENT THAN HAHN ECHO !!!!!!!!!!!!!!!!!*/
		#if 0
		W1=W1%cce1(hamplus1,hamminus1, Uplus1, Uminus1, t, A1[0], omega1[0]);
		#endif
	//cout<<i<<"\n";
	j=i+1;

	while(j<N){
		
		if((as75_loc.row(j)-as75_loc.row(i)).norm()<R){
			random=((double) rand() / (RAND_MAX));
			if(random<percent){
	//cout<<"i: "<<i<<" j: "<<j<<"\n";
	A1.push_back(A(as75_loc.row(j),NV_loc,B_orient,l0,z0,a0,"75As"));
	
	omega1.push_back(omega(magn_field));
	#if 0
	for(int ik=0;ik<A1.size();ik++){
		cout<<"A: "<<A1[ik]<<"\t";
	}
	cout<<"\n";
	#endif

	b12=b(as75_loc.row(i),as75_loc.row(j),B_orient,"75As");
	//cout<<b12<<"\n";
	W2_75As=W2_75As%cce2(hamplus2, hamminus2, Uplus2, Uminus2, t, A1, omega1, b12);
	//cout<<"ROSJA2"<<"\n";
	//Wan=Wan%cce2_anal(t, A1, omega1, b12);

	A1.pop_back();

	omega1.pop_back();
	neighb++;
	}
}
	j++;
	}
	A1.pop_back();
	omega1.pop_back();
	//cout<<"Neighbours included: "<<neighb<<endl;
	i++;
	}
	for(int i=0;i<W2_75As.size();i++){
		W2_75As[i]=pow(abs(W2_75As[i]),1/percent);
	}
	//W2_75As=pow(abs(W2_75As),1/percent);
	W2=W2%W1%W2_75As;
	cout<<"75 As calculated."<<endl;
	//cout<<"W1: "<<"\t"<<"W2: "<<"\t"<<"\t"<<"Wan: "<<"Wan_py: "<<"\n";
	
	for(int i=0;i<t.size();i++){
		cout<<W1[i]<<"\t"<<W2[i]<<"\t"<<"\n";
		//cout<<t[i]<<"\t"<<Wan[i]<<"\t"<<Wan_py(i,1)<<"\n";
	}
	
	
}