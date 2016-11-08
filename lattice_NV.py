# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 14:20:50 2015

@author: Admin
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
def GaAs_lattice(r_sph,z0,a0):
    #there are 4 atoms inside each fcc cells and 2 of them build one unit cell. 
#    N=100*nC13/1.1
    pa69Ga=0.604
    pa71Ga=0.302
    
    #basis vectors=basis[i,:]
    basis=np.array([np.zeros(3),0.25*np.ones(3)])
    #primitive cell vectors=pcvect[i,:]
    pcvect=0.5*np.array([[0,1,1],[1,0,1],[1,1,0]])
    #redefine what is normal vector: nx,ny,nz in the algebraic base of 
    #primitive cell vectors whereas ni have length of lattice constant
#    nx=np.array([-1,1,1]) #corresponds to (n,l,m) values 
#    ny=np.array([1,-1,1])
#    nz=np.array([1,1,-1])
    As_loc=[]
    a69Ga_loc=[]
    a71Ga_loc=[]
    N=int((4*r_sph)/a0) #defines a maximal index
    M=int(z0/a0)
#    NV_loc=a0*(int(N/2)*pcvect[0,:]+int(N/2)*pcvect[1,:]+int(N/2)*pcvect[1,:]+basis[0,:])
#    NV_orient=(0.25*np.ones(3)) #vacancy located in the neighbouring primitive unit
#    total=(2*(N)**3)
    nAs=0
    na69Ga=0
    na71Ga=0
#    i=0
    for n in range(0,N):
        for l in range(0,N):
            for m in range(0,N):
#                new=0
                #p1=np.random.rand()
                #if p1<=0.011:
                
#               print 'p1'
                z=n*pcvect[0,:]+l*pcvect[1,:]+m*pcvect[2,:]+basis[0,:]
#                print z
                if a0*z[2]-z0<0:
                    nAs+=1
                    As_loc.append(a0*z)
                p1=np.random.rand()
                if p1<=pa69Ga:
                    
#                    print 'p2'
                    z=n*pcvect[0,:]+l*pcvect[1,:]+m*pcvect[2,:]+basis[1,:]
                    if abs(a0*z[2])-z0<0:
                        na69Ga+=1
                        a69Ga_loc.append(a0*z)
                if p1>pa69Ga and p1<=pa69Ga+pa71Ga:
                    

                    z=n*pcvect[0,:]+l*pcvect[1,:]+m*pcvect[2,:]+basis[1,:]
                    if abs(a0*z[2])-z0<0:
                        na71Ga+=1
                        a71Ga_loc.append(a0*z)
                
    As_loc=np.array(As_loc)                                                    
    a69Ga_loc=np.array(a69Ga_loc)
    a71Ga_loc=np.array(a71Ga_loc)
    #C-13 atoms location
    
    print('Nr of 69Ga: '+str(na69Ga)+' 71Ga: '+str(na71Ga)+ ' As: '+str(nAs))
#    c13_loc_arr=np.zeros((nC13,3))
#    c13_loc=np.array(c13_loc)
#    i=0
#    j=0
#    while i<nC13:
#        if j!=1:
#            c13_loc1=c13_loc[i,:]
#        problem1=(3/norm(NV_orient)**2)*np.array([np.dot((c13_loc1-c13_loc_arr[k,:]),NV_orient)**2 for k in range(nC13)])-np.array([norm(c13_loc_arr[j,:]-c13_loc1)**2 for j in range(nC13)])
#        problem2=(3/norm(NV_orient)**2)*np.dot((c13_loc1-NV_loc),NV_orient)**2-norm(c13_loc1-NV_loc)**2
#        problem3=[norm(c13_loc_arr[m,:]-NV_loc) for m in range(nC13)]
#        #problem4=
#        rivrjv=np.where(problem3==norm(c13_loc1-NV_loc))#find indices for which ||r_iv||=||r_jv||
#        #print rivrjv
#        rij=np.array([(c13_loc1-c13_loc_arr[elem,:]) for elem in rivrjv]) # for those elements check if also 4|r_ij.NV_or|==norm(r_ij)
#        #print rij       
#        #print rij[0,0]
#        normrij=norm(rij,axis=1)
#        rijdotNV=np.array([np.abs(np.dot(rij[elem1,:],NV_orient)) for elem1 in range(rij.shape[0])])
#        problem4=np.where(normrij==4*rijdotNV)
#        #print len(problem4[0])
#        
#        if len(problem1[problem1==0])==0 and problem2!=0 and len(problem4[0])==0:
#            c13_loc_arr[i,:]=c13_loc1
#            i+=1
#            j=0
#        else:
#            prob=np.random.rand();
#            if prob<=0.5:
#                c13_loc1=a0*np.array(N*np.random.rand()*pcvect[0,:]+N*np.random.rand()*pcvect[1,:]+N*np.random.rand()*pcvect[1,:]+basis[0,:])
#            elif prob>0.5:
#                c13_loc1=a0*np.array(N*np.random.rand()*pcvect[0,:]+N*np.random.rand()*pcvect[1,:]+N*np.random.rand()*pcvect[1,:]+basis[1,:])
##            print i, c13_loc1
#            j=1
##            i+=1
#    print(len(c13_loc_arr))
#      
#    #c13_loc=a0*c13_loc
#    #print norm(c13_loc,axis=1)
#    #NV centre location
    
#    print np.tile(NV_loc,(5,1))
    a69Gasorter=np.argsort(np.linalg.norm(a69Ga_loc-np.tile([r_sph,r_sph,z0/2],(len(a69Ga_loc),1)),axis=1))
    #print c13_loc
    a69Ga_loc=a69Ga_loc[a69Gasorter,:]
    
    a71Gasorter=np.argsort(np.linalg.norm(a71Ga_loc-np.tile([r_sph,r_sph,z0/2],(len(a71Ga_loc),1)),axis=1))
    #print c13_loc
    a71Ga_loc=a71Ga_loc[a71Gasorter,:]
    
    Assorter=np.argsort(np.linalg.norm(As_loc-np.tile([r_sph,r_sph,z0/2],(len(As_loc),1)),axis=1))
#    #print c13_loc
    As_loc=As_loc[Assorter,:]
    
#    print "Condition set in nanometers!!"
#   extracting elements inside a cylinder constraint by base radius l and height z0
#    condition=np.where(np.abs(a69Ga_loc[:,2]-np.tile(z0,(len(a69Ga_loc),1)))>0)
#    a69Ga_loc=a69Ga_loc[condition[0][:],:]
##        
#    condition=np.where(a71Ga_loc[:,2]-np.tile(z0,(len(a71Ga_loc),1))>0)
#    a71Ga_loc=a71Ga_loc[condition[0][:],:]
##    
#    condition=np.where(As_loc[:,2]-np.tile(z0,(len(As_loc),1))>0)
#    As_loc=As_loc[condition[0][:],:]    
#    print(condition[0][:])
#    c13_loc1=np.zeros((len(condition[0][:]),3))
#    for i in range(len(condition[0][:])):
#        c13_loc1[i,:]=c13_loc[condition[0][i],:]
    #print c13_loc
    
    return a69Ga_loc,a71Ga_loc,As_loc#,total
    
    
    
        
if __name__ == '__main__':
    #defines nr of 13-C atoms (pseudo-1/2 spins) spread in the diamond lattice
#    nC13=1
    #lattice constant
    a0=0.565
    l0=70
    z0=10
    a69Ga_loc,a71Ga_loc,As_loc=GaAs_lattice(l0,z0,a0)
#    print np.linalg.norm(a71Ga_loc-np.tile([l,l,z0/2],(len(a71Ga_loc),1)),axis=1)
    """import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a69Ga_loc[:,0],a69Ga_loc[:,1],a69Ga_loc[:,2],zdir='z',c='b')
    ax.scatter(a71Ga_loc[:,0],a71Ga_loc[:,1],a71Ga_loc[:,2],zdir='z',c='r')
    ax.scatter(As_loc[:,0],As_loc[:,1],As_loc[:,2],zdir='z',c='g')
    plt.show()"""
#        print c13_loc
#        print NV_orient
    ga69=open("69Ga_dist_z0_"+str(int(z0))+"_l0_"+str(int(l0))+".dat","w")
    for row in a69Ga_loc:
        ga69.write(str(row[0])+"\t"+str(row[1])+"\t"+str(row[2])+"\n")
    ga69.close()
    ga71=open("71Ga_dist_z0_"+str(int(z0))+"_l0_"+str(int(l0))+".dat","w")
    for row in a71Ga_loc:
        ga71.write(str(row[0])+"\t"+str(row[1])+"\t"+str(row[2])+"\n")
    ga71.close()
    as75=open("75As_dist_z0_"+str(int(z0))+"_l0_"+str(int(l0))+".dat","w")
    for row in As_loc:
        as75.write(str(row[0])+"\t"+str(row[1])+"\t"+str(row[2])+"\n")
    as75.close()
else:
    print('Importing lattice generator')