      subroutine pirock(neqn,npdes,t,tend,h,y,f,fd2,fa,fr,fw,
     &    atol,rtol,frjac,ijac,work,iwork,idid)
c ----------------------------------------------------------
c   
c    Authors: A. Abdulle and G. Vilmart
c
c    Version of 8 October 2012
c    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
c    PLEASE CITE THE FOLLOWING PAPERS WHEN USING PIROCK:
c
c    [ ] A. Abdulle & G. Vilmart
c        PIROCK: a swiss-knife partitioned implicit-explicit 
c        orthogonal Runge-Kutta Chebyshev integrator for stiff 
c        diffusion-advection-reaction problems with or without noise,
c        Journal of Computational Physics 242 (2013), 869--888.
c        doi: http://dx.doi.org/10.1016/j.jcp.2013.02.009
c        
c      This code is a modification of the code ROCK2 (A. Abdulle, 2002)
c
c    [ ] A. Abdulle & A.A. Medovikov
c        Second order Chebyshev methods based on
c        orthogonal polynomials
c        Numer. Math.  90, no. 1 (2001), 1--18.
c
c     Input parameters  
c     ----------------  
c     NEQN:       Number of differential equations of the system 
c                 (integer).
c
c     T:          Initial point of integration (double precision).
c
c     TEND:       End of the interval of integration,
c                 may be less than t (double precision).
c
c     H:          Initial step size guess 
c                 (usually between 1d-4 and 1d-6).
c
c     Y(NEQN):    Initial value of the solution 
c                 (double precision array of length neqn).
c
c     F:          Name (external) of subroutine computing the value 
c                 of f(x,y). Must have the form
c
c                   subroutine f(neqn,t,y,dy)
c                   double precision y(neqn),dy(neqn)
c                   integer neqn
c                   dy(1)=...
c                   ...
c                   dy(neqn)=...
c                   return
c                   end 
c         
c                 Implementation:   
c                 for stability issues when the problem 
c                 is originating from parabolic PDEs, transforming 
c                 inhomogeneous boundary conditions in homogeneous ones
c                 (by adding the appropriate function to the right-hand side)
c                 may increase the performance of the code.
c
c                 
c     ATOL(*) :   Absolute and relative error tolerances 
c     RTOL(*)     can be both scalar (double precision)
c                 or vectors of length neqn (double precision).
c
c     RHODIFF:    Name (external) of a function (double precision) 
c                 giving the spectral radius of the Jacobian 
c                 matrix  of f at (t,y). Must have the form
c                 
c                   double precision function rhodiff(neqn,t,y)
c                   double precision y(neqn),t
c                   integer neqn
c                   ...
c                   rhodiff=... 
c                   return
c                   end
c               
c                 N.b. Gerschgorin's theorem can be helpful. If the
c                 Jacobian is known to be constant it should be 
c                 specified by setting iwork(2)=1 (see below).
c
c                 ROCK2 can also compute this estimate. In that 
c                 case, provide a dummy function rhodiff(neqn,t,y) and 
c                 set iwork(1)=0 (see below).
c
c                 If it is possible to give an estimate of 
c                 the spectral radius, it should be preferred to
c                 the estimate computed internally by ROCK2.
c
c     IWORK(*):   Integer array of length 25 that gives information 
c                 on how the problem is to be solved and communicates 
c                 statistics about the integration process.
c               
c     IWORK(1):   =0 ROCK2 attempts to compute the spectral radius 
c                    internally. Define a dummy function
c
c                    double precision function rhodiff(neqn,t,y)
c                    double precision y(neqn),t
c                    integer neqn
c                    rhodiff=0.d0 
c                    return
c                    end
c
c                 =1 RHO returns an upper bound of the spectral 
c                    radius  of the Jacobian matrix of f at (t,y).
c    
c     IWORK(2):   =0 The Jacobian is not constant.                  
c                 =1 The Jacobian is constant,
c                    the function rho is called only once.
c
c     IWORK(3):   =0 Return and solution at tend
c                 =1 the code returns after each step t_i chosen
c                    automatically between [t,tend] (solution
c                    at t_i is in y(*) ).
c                    To continue call ROCK2 again without changing
c                    any arguments.
c
c     IWORK(4):   =0 Atol and rtol are scalar.
c                 =1 Atol and rtol are array of length neqn.
c
c     WORK(*) :     Workspace of length 5*neqn if iwork(1)=0,
c                   otherwise of length 4*neqn.
c                   Work(1),..,work(4*neqn) serve as
c                   working space for the solution of
c                   the ode.
c                   Work(4*neqn+1),..,work(5*neqn)
c                   serve as working space for the
c                   internal computation of the 
c                   spectral radius of the Jacobian.
c                  
c     IDID:         Report on successfulness upon return
c                   (integer).
c
c
c     Output parameters 
c     -----------------
c     T:          T-value for which the solution has been computed
c                 (after successful return t=tend).
c
c     Y(NEQN):    Numerical solution at tend.
c
c     IDID:       Reports what happened upon return
c
c     IDID        =1 Successful computation t=tend.
c                 =2 Successful computation of one step
c                    to continue call ROCK2 again without
c                    altering any arguments.
c                 =-1 Invalid input parameters.
c                 =-2 Stepsize becomes to small.
c                 =-3 The method used in ROCK2 to estimate 
c                     the spectral radius did not converge.
c                 =-4 quasi-newton method failed to converge.
c                 =-5 LU decomposition error in the quasi-newton method.
c
c     IWORK(5)    =Number of function evaluations (diffusion).
c     IWORK(6)    =Number of steps.
c     IWORK(7)    =Number of accepted steps.
c     IWORK(8)    =Number of rejected steps.
c     IWORK(9)    =Number of evaluations of f used
c                  to estimate the spectral radius
c                  (equal to zero if iwork(1)=1).
c     IWORK(10)   =Maximum number of stages used.
c     IWORK(11)   =Maximum value of the estimated 
c                  bound for the spectral radius 
c                  (rounded to the nearest integer).
c     IWORK(12)   =Minimum value of the estimated  
c                  bound for the spectral radius 
c                  (rounded to the nearest integer).
c     IWORK(13)   =Maximum number of iterations used 
c                  in the quasi-Newton method.
c     IWORK(14)   =Maximum value of the estimated 
c                  bound for the spectral radius of advection
c     IWORK(15)   =Minimum value of the estimated 
c                  bound for the spectral radius of advection
c     IWORK(16)    =Number of function evaluations (advection).
c     IWORK(17)    =Number of function evaluations (reaction).
c     IWORK(18)    =Number of function evaluations (jacobian reaction).
c
c
c     IWORK(19)   =2 Stepsize control with memory
c                 =1 Stepsize control without memory
c                 =0 Constant stepsize
c     IWORK(20)   =1 Enable F_A (advection or nonstiff reaction)
c     IWORK(21)   =1 Enable F_R (stiff reaction)
c     IWORK(22)   =1 Enable F_W (noise)
c     IWORK(23)   =1 Verbose (write stepsizes, errors)
c     IWORK(24)   =1 Non-symmetric diffusion operator
c                       
c   
c    Caution:     The variable UROUND (the rounding unit) is set to 
c    -------      1.0d-16 and may depends on the machines.
c-------------------------------------------------------------------
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c         Numerical method
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
c      The nearly optimal stability polynomial is computed
c      as a product:  R_s(z)=P_{s-2}(z)*w(z).
c      We realize this polynomial as a Runge-Kutta method
c      with a three-term recurrence formula for the first
c      s-2 stages (with P_{s-2}(z) as stability polynomial)
c      and a 2-stage finishing procedure (with w(z) as 
c      stability polynomial).
c
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c         Stability functions and three-term recurrence relation.
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
c     
c     The stability functions: R_j(z)=P_{j-2}(z)  (internal j<=ms-2)
c                             R_{ms}=P_{s-2}(z)*w(z) ( absolute) 
c                             w(z)=1+sigma*z+tau*z^2 
c                 
c     P_j(z) orthogonal with respect to w(z)^2/sqrt{1-x^2}
c     
c     Recurrence formula: 
c     
c       P_j(z)=(a_j*z-b_j)*P_{j-1}(z)-c_j*P_{j-2}(z) 
c                   j=1..ms-2 b_1=-1,c_1=0
c
c     Normalization: P_(0)=1  =>b_{j}=-(1+c_{j}) 
c
c     Runge-Kutta formula:
c
c     g_j(z)=a_{j}*f(g_{j-1})-b_{j}*g_{j-1}-c_{j}*g_{j-2} j=1..ms-2 
c
c     Data (rec. param.): rec(i)= a_1,a_2,c_2,a_3,c_3,.,a_(ms-2),
c                          c_(ms-2)  for ms=1,3,5,..  
c
c     The two-stage finishing procedure:
c
c     g_{s-1}=g_{s-2} + h*sigma*f(g_{s-2}
c     ge_{s}=g_{s-1} + h*sigma*f(g_{s-1}
c     g_{s}=ge_{s} -h*sigma*(1-(sigma/tau)*(f(g_{s-1}-f(g_{s-2})
c
c     Embedded method: ge_{s}
c
c     Datas (finish. proced.): fp1(ms) =sigma  
c                              fp2(ms) =-sigma(1-(sigma/tau) 
c                              ms=1,3,5,.. 
c
c     Chosen degrees: s=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
c     20,21,22,24,26,28,30,32,35,38,41,45,49,53,58,63,68,74,80,87,95,
c     104,114,125,137,150,165,182,200  ms=s-2
c
c------------------------------------------------------------------
c
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***          
c             Declarations 
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
c
      double precision y(neqn),work(*),atol(*),rtol(*),
     & t,tend,h,uround,
     & recf(4476),fp1(46),fp2(46),  
     & recf2(184),recalph(46),frjac(*)
      integer iwork(25),ms(46),neqn,i,n1,n2,n3,n4,n5,n6,n7,n8,
     & n9,n10,n11,n12,n13,n14,n15,ntol,idid,npdes,ijac(*)
      logical arret
      external f,fd2,fa,fr,fw
c -------- Uround: smallest number satisfying 1.d0+uround>1.d0
      data uround/1.0d-16/ 
c             
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
c             Data of the stability polynomials
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
c included file 'rectp.f' with coefficients
      include 'rectp.f'

c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c             Initializations
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
      arret=.false.
c --- check config
		  if (iwork(22).eq.1) iwork(19)=0
      if (iwork(19).eq.0) iwork(4)=0
		  if (iwork(20).eq.0.and.iwork(21).eq.0.and.iwork(22).eq.0
     &   .and.iwork(19).eq.1) iwork(19)=2
c -------- Prepare the entry-points for the arrays in work.--------     
      n1=1
      n2=n1+neqn
      n3=n2+neqn
      n4=n3+neqn
			n5=n4+neqn
			n6=n5+neqn
			n7=n6+neqn
			n8=n7+neqn
			n9=n8+neqn
			n10=n9+neqn
			n11=n10+neqn
			n12=n11+neqn
			n13=n12+neqn
			n14=n13+neqn
			n15=n14+neqn
c -------- Test the initial step size and tolerances.--------   
c gilles modif to avoid round off error bug 
      if (abs(h-abs(tend-t)).le.1.d-12) then
			   h=abs(tend-t)
			else
      if (h.gt.abs(tend-t)) then
        write(6,*)'initial step is longer than the integration interval'
     &   ,h,abs(tend-t),h-abs(tend-t)
        arret=.true.
        idid=-1
        return
      end if 
			end if 
			if (iwork(19).gt.0) then
      if (h.lt.10.d0*uround) then
        write(6,*)' initial step-size is too small'
        idid=-1
        arret=.true.
        return
      end if 
      if (iwork(4).eq.0) then
        ntol=0
        if (atol(1).le.0.d0.or.rtol(1).le.10.d0*uround) then
          write(6,*) 'tolerances are too small'
          arret=.true.
          idid=-1
          return
        end if
      else
        ntol=1
        do i=1,neqn
        if (atol(i).le.0.d0.or.rtol(i).le.10.d0*uround) then
          write(6,*) 'tolerances are too small'
          arret=.true.
          idid=-1
          return
        end if
        end do
      end if
			else
			ntol=0
			end if
c -------- Call to the core integrator. -----------
      call rockcore(neqn,npdes,t,tend,h,y,work(n15),f,fd2,fa,fr,fw,
     & work,work(n1),work(n2),work(n3),work(n4),ms,atol,rtol,ntol,recf,
     & fp1,fp2,iwork,arret,uround,idid,work(n5),recf2,recalph,
     & work(n6),work(n7),work(n8),work(n9),work(n10),work(n11),
     & work(n12),work(n13),work(n14),frjac,ijac)
      return
      end
c ----------------------------------------------
c     End of subroutine ROCK2.
c ----------------------------------------------
c
      subroutine rockcore(neqn,npdes,t,tend,h,y,ye,f,fd2,fa,fr,fw,
     & work,yn,fn,yjm1,yjm2,ms,atol,rtol,ntol,recf,fp1,fp2,iwork,
     & arret,uround,idid,yks,recf2,recalph,yjm3,yjm4,yjm5,y2,
     & fnc,yrk0,ytmp,yerrA,yerrR,frjac,ijac)
c ----------------------------------------------
c    Core integrator for ROCK2.
c ---------------------------------------------- 
c             Declarations
c ----------------------------------------------
       double precision y(*),ye(*),yn(*),fn(*),work(*),yjm1(neqn),
     & yjm2(neqn),recf(4476),atol(*),rtol(*),fp1(46),fp2(46),
     & err,errp,tend,h,hnew,hp,facmax,fac,facp,rhodiff,rhoadv,uround,
     & recalph(46),eigmax,eigmaxadv,t,te,ta,told,tmp,facd,
     & recf2(184),yks(neqn),beta,yjm3(neqn),yjm4(neqn),yjm5(neqn),
     & ytmp(neqn),fnc(neqn),y2(neqn),yrk0(neqn),
     & yerrA(neqn),yerrR(neqn),errD,errD2,errA,errR,frjac(*)
       integer ms(46),mp(2),iwork(25),neqn,mdeg,i,ntol,
     & nrho,mdego,nrej,idid,npdes,ijac(*),nell,nrejfac
       logical last,reject,arret
       external f,fd2,fa,fr,fw
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c             Initializations
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
      do i=5,13
      iwork(i)=0
			end do
      facmax=5.d0
			facd=0.8d0
			nrejfac=0
      nrej=0
      told=0.d0
      mdego=0
      last=.false.
      reject=.false. 
      hp=h
      err=0.d0
			errD=0.d0
			errD2=0.d0
			errA=0.d0
			errR=0.d0
      nrho=0
      idid=1
			nell=0
c for compensated summation
      do i=1,neqn
			ye(i)=0.d0
			end do
			te=0.d0
c -------- Initialization of the integration step.--------   
10    do i=1,neqn
       yn(i)=y(i)
      end do
      call f(neqn,t,yn,fn)
      iwork(5)=iwork(5)+1
      errp=err
c -------- Step size is adjusted.--------
20    if(1.1d0*h.ge.abs(tend-t)) then
        h=abs(tend-t)
        last=.true.
      end if
      if (iwork(19).gt.0.and.h.lt.10.d0*uround) then
        write(6,*)' tolerances are too small'
        idid=-2
        arret=.true.
        return
      end if
c -------- Spectral radius.--------
      if (nrho.eq.0) then
        if (iwork(6).eq.0.or.iwork(2).eq.0) then
c ------- Computed externally by rhodiff.--------
          if (iwork(1).eq.1) then
            eigmax=rhodiff(neqn,t,yn)
            if (idnint(eigmax).gt.iwork(11)) 
     &         iwork(11)=idnint(eigmax)
            if (iwork(6).eq.0) iwork(12)=iwork(11)
            if (idnint(eigmax).lt.iwork(12)) 
     &         iwork(12)=idnint(eigmax)
						if (iwork(20).eq.1) then
						eigmaxadv=rhoadv(neqn,t,yn)
						if (idnint(eigmaxadv).gt.iwork(14)) 
     &         iwork(14)=idnint(eigmaxadv)
            if (iwork(6).eq.0) iwork(15)=iwork(14)
            if (idnint(eigmaxadv).lt.iwork(15)) 
     &         iwork(15)=idnint(eigmaxadv)
						end if
c ------- Computed internally by rocktrho.--------  
                             else
            call rocktrho(neqn,t,y,f,yn,fn,work,yjm1,
     &                   yjm2,eigmax,uround,idid,iwork)
            if (idnint(eigmax).gt.iwork(11)) 
     &         iwork(11)=idnint(eigmax)
            if (iwork(6).eq.0) iwork(12)=iwork(11)
            if (idnint(eigmax).lt.iwork(12)) 
     &         iwork(12)=idnint(eigmax)
		        if (iwork(20).eq.1) then
						call rocktrho(neqn,t,y,fa,yn,fn,
     &                   work,yjm1,yjm2,eigmaxadv,uround,idid,iwork)
		        if (idnint(eigmax).gt.iwork(14)) 
     &         iwork(14)=idnint(eigmax)
            if (iwork(6).eq.0) iwork(15)=iwork(14)
            if (idnint(eigmax).lt.iwork(15)) 
     &         iwork(15)=idnint(eigmax)
		        end if
          end if
        end if
      end if
c -------- The number of stages.--------
      mdeg=sqrt((1.5d0+h*eigmax)/0.811d0)+1
			mdeg=max(3,mdeg)
      if (mdeg.gt.200) then
        h=0.8d0*(200.d0**2*0.811d0-1.5d0)/eigmax
        mdeg=200
        last=.false.
      end if
			if (eigmax*h.le.2.5d0) then
			nell=-1
c stepsize CFL for the advection terms
			elseif (iwork(20).eq.1) then
			tmp=(0.07696d0*mdeg+1.878d0)/max(eigmaxadv,0.1d0)
 	    if (tmp.lt.0.5d0*h.or.mdeg.le.4) then
      nell=1
c change facd
			if (mdeg.gt.4) facd=min(0.4d0,facd)
      mdeg=sqrt((1.5d0+h*eigmax)/0.432d0)+1
			mdeg=max(3,mdeg)
c			tmp=(0.5321d0*mdeg+0.4996d0)/max(eigmaxadv,0.1d0)
c      if (tmp.lt.h)	then
c			h=tmp
c			mdeg=sqrt((1.5d0+h*eigmax)/0.432d0)+1
c			end if
			if (mdeg.gt.200) then
        h=0.8d0*(200.d0**2*0.432d0-1.5d0)/eigmax
        mdeg=200
        last=.false.
      end if
			else
			nell=0
			end if
			else
			nell=0
			end if
c non symmetric diffusion
			if (iwork(24).eq.1) then
			nell=1
			mdeg=sqrt((1.5d0+h*eigmax)/0.432d0)+1
			end if
c
      mdeg=max(mdeg,3)-2
      if (mdeg.ne.mdego) then
        call mdegr(mdeg,mp,ms)
      end if
      if (mdeg+2.gt.iwork(10)) iwork(10)=mdeg+2
			if (iwork(23).eq.1) 
     & write (6,*) '--t',t,'h',h,'mdeg',mdeg+2,'ell',nell,'facd',facd
c -------- Computation of an integration step.--------
      if (nell.ge.0) then
      call rtstep(neqn,t,h,y,ye,f,yn,fn,yjm1,
     & yjm2,mdeg,mp,errD,atol,rtol,ntol,recf,fp1,fp2,
     & recf2,recalph,yks,beta,nell,iwork)
c --- PIROCK: partitioned method
c avoid this computation if the step is rejected
      if ((iwork(19).eq.0.or.errD.lt.1.d0).and.
     &   (iwork(20).eq.1.or.iwork(21).eq.1
     &   .or.iwork(22).eq.1).or.iwork(24).eq.1) then
			call rkstep(neqn,npdes,t,h,y,ye,f,fd2,fa,fr,fw,yks,
     &yjm1,yjm2,yjm3,yjm4,yjm5,yrk0,yn,ytmp,y2,yerrA,yerrR,fnc,
     &errA,errR,errD2,atol,rtol,ntol,beta,frjac,ijac,iwork,idid,nell)	
			if (iwork(20).eq.1) iwork(16)=iwork(16)+3
			iwork(5)=iwork(5)+2
			end if
			else
      call rkstep0(neqn,npdes,t,h,y,ye,f,fd2,fa,fr,fw,fn,
     &yjm1,yjm2,yjm3,yjm4,yjm5,yrk0,yn,ytmp,y2,yerrA,yerrR,fnc,
     &errD,errD2,errA,errR,atol,rtol,ntol,frjac,ijac,iwork,idid,nell)
			end if
      if (iwork(23).eq.1) 
     & write (6,*) 'errD',errD,'errD2',errD2,'errA',errA,'errR',errR
			err=dmax1(dmax1(dmax1(errD,errD2),errA),errR)
c ---
      mdego=mdeg
      iwork(6)=iwork(6)+1 
      if (nell.ge.0) iwork(5)=iwork(5)+mdeg+1   
			if (iwork(20).eq.1.or.iwork(21).eq.1.or.iwork(22).eq.1) then
			if (nell.le.0) iwork(5)=iwork(5)+2  
			if (nell.gt.0) iwork(5)=iwork(5)+1 
			end if
c -------- Error control procedure.--------
      if (iwork(19).gt.0) then
c detection of stiff splittings
			if ((iwork(20).eq.1.or.iwork(21).eq.1)
     &               .and.mod(iwork(6),100).eq.0) then
		  if (nrejfac.ge.10) facd=facd*0.8d0
			if (nrejfac.eq.0) facd=facd*1.02d0
			facd=dmin1(0.8d0,dmax1(0.4d0,facd))
c			write (6,*) t,facd,nrejfac
			nrejfac=0
			end if
c
      fac=sqrt(1.d0/err)
c stepsize selection (memory)
      if (iwork(19).eq.2.and.errp.ne.0.d0.and..not.reject.
     &      and.(errD.ge.(10.d0*dmax1(errA,errR)).or.nell.lt.0)) then
       facp=sqrt(errp)*fac**2*(h/hp)
c
       fac=dmin1(fac,facp)
      end if
      if (reject) then
        facmax=1.d0		
      end if 
      fac=dmin1(facmax,dmax1(0.1d0,facd*fac)) 
c standard stepsize selection
       hnew=h*fac
			 else 
			 hnew=h
			 end if
c -------- Accepted step.--------
      if (err.lt.1.d0.or.iwork(19).eq.0) then
cc gilles solution (for random problem) zzz
			write (17,*) t+h,y(1),y(2)
c			write (17,*) t+h,y(40000-1),y(40000)
        iwork(7)=iwork(7)+1
        facmax=2.d0
c compensated summation (t=t+h)
        ta=t
        te=te+h
				t=t+te
				te=te+(ta-t)
c
        if (reject) then
          hnew=dmin1(hnew,h)
          if (tend.lt.t)  hnew=dmax1(hnew,h)
          reject= .false.
          nrej=0
        end if
        hp=h
        h=hnew
        nrho=nrho+1
        nrho=mod(nrho+1,25)
        if (last) then 
          return
        elseif (iwork(3).eq.1) then
          if (idid.eq.1) idid=2
          if(1.1d0*h.ge.abs(tend-t)) h=dabs(tend-t)
          return
        else
          goto 10
        end if
                   else            
c -------- Rejected step.--------
        if (iwork(23).eq.1) 
     &  write (6,*) 'rejected step at t=',t,'h=',h,
     &   'errD',errD,'errD2',errD2,'errA',errA,'errR',errR
        iwork(8)=iwork(8)+1
				nrejfac=nrejfac+1
        reject= .true.
        last=.false.
        h= 0.8d0*hnew
        if (iwork(6).eq.0) h=0.1d0*h
        if (told.eq.t) then 
          nrej=nrej+1
          if (nrej.eq.10) h=1.0d-5
        end if
        told=t   
c -------The spectral radius is recomputed.--------
c        after a step failure
        if (nrho.ne.0) then 
          nrho=0
                       else
          nrho=1
        end if
        goto 20
      end if        
      return      
      end
c ----------------------------------------------
c     End of subroutine rockcore.
c ----------------------------------------------  
c           
      subroutine rtstep(neqn,t,h,y,ye,f,yn,fn,yjm1,yjm2,
     & mdeg,mp,err,atol,rtol,ntol,recf,fp1,fp2,
     & recf2,recalph,yks,beta,nell,iwork)
c ----------------------------------------------
c  Solut. at t+h by an explicit (mdeg+2)-stages formula.
c-----------------------------------------------
c             Declarations
c-----------------------------------------------
       double precision y(neqn),yn(neqn),fn(neqn),yjm1(neqn),
     & yjm2(neqn),recf(4476),atol(*),rtol(*),
     & fp1(46),fp2(46),recalph(46),err,t,h,ci1,ci2,ci3,ye(neqn),
     & temp1,temp2,temp3,ato,rto,recf2(184),yks(neqn),beta,hn
       integer mp(2),neqn,mdeg,mr,mz,i,j,ntol,iwork(*)
       external f
			 double precision anor
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c             Initialisations
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***       
      err=0.d0
      mz=mp(1)
      mr=mp(2)
			hn=h
			if (nell.ge.1) hn=h*recalph(mz)
c -------- First stage.--------
      temp1=hn*recf(mr)
      ci1=temp1
      ci2=temp1
      ci3=0.d0
      do i=1,neqn
c compensated summation
        yjm2(i)=temp1*fn(i)
				ye(i)=ye(i)+yjm2(i)
			  yjm1(i)=yn(i)+ye(i)
				ye(i)=ye(i)+(yn(i)-yjm1(i))
        if (mdeg.lt.2) y(i)=yjm1(i)
      end do
c --------- Stages for j=2..mdeg.--------
      do i=2,mdeg
        temp1=hn*recf(mr+2*(i-2)+1)
        temp3=-recf(mr+2*(i-2)+2)
        temp2=1.d0-temp3
        call f(neqn,t+ci1,yjm1,y)
        ci1=temp1+temp2*ci2+temp3*ci3
        do j=1,neqn
c compensated summation (yjm2(j)<-yjm1(j)-yjm2(j))
          yjm2(j)=temp1*y(j)-temp3*yjm2(j)
					ye(j)=ye(j)+yjm2(j)
					y(j)=yjm1(j)+ye(j)
					ye(j)=ye(j)+(yjm1(j)-y(j))
c -------- Shift the value "y" for the next stage.--------
          if (i.lt.mdeg) then
            yjm1(j)=y(j)
          end if
        end do
        ci3=ci2
        ci2=ci1
      end do
c --- PIROCK: compute two additional orthog. polynomials.
      if (iwork(20).eq.1.or.iwork(21).eq.1
     &    .or.iwork(22).eq.1.or.iwork(24).eq.1) then
      temp1=hn*recf2((mz-1)*4+1)
			temp3=-recf2((mz-1)*4+2)
			temp2=1.d0-temp3
			call f(neqn,t+ci1,y,yks)
			beta=temp1+temp2*ci2+temp3*ci3
      do j=1,neqn
				yjm2(j)=yjm1(j)
				yjm1(j)=y(j)
			  yks(j)=temp1*yks(j)+temp2*yjm1(j)+temp3*yjm2(j)
			  yjm2(j)=yjm1(j)
				yjm1(j)=yks(j)
			end do
			if (nell.eq.0) then
      temp1=hn*recf2((mz-1)*4+3)
			temp3=-recf2((mz-1)*4+4)
			temp2=1.d0-temp3
			call f(neqn,t+beta,yjm1,yks)
			beta=temp1+temp2*beta+temp3*ci2
      do j=1,neqn
			  yks(j)=temp1*yks(j)+temp2*yjm1(j)+temp3*yjm2(j)
			end do
			end if
			end if
c -------- The two-stage finishing procedure.--------
      temp1=h*fp1(mz)
      temp2=h*fp2(mz)
			if (nell.ge.1) then
      amu=recalph(mz)
			temp1=0.5d0*(h-hn)+hn*fp1(mz)
			temp3=0.5d0*(amu-1.d0)**2+2.d0*fp1(mz)*amu
     &     *(1.d0-amu)+amu**2*(fp2(mz)+fp1(mz))*fp1(mz)
			temp2=-temp1*(1.d0-temp3*(h/temp1)**2)
			end if
      call f(neqn,t+ci1,y,yjm2)
      do j=1,neqn
				ye(j)=ye(j)+temp1*yjm2(j)
			  yjm1(j)=y(j)+ye(j)
				ye(j)=ye(j)+(y(j)-yjm1(j))
      end do
      ci1=ci1+temp1
      call f(neqn,t+ci1,yjm1,y)
c -------- Atol and rtol are scalar.--------
      if (ntol.eq.0) then
        ato=atol(1)
        rto=rtol(1)
        do j=1,neqn
          temp3=temp2*(y(j)-yjm2(j))
				   ye(j)=ye(j)+temp1*y(j)+temp3
			     y(j)=yjm1(j)+ye(j)
				   ye(j)=ye(j)+(yjm1(j)-y(j))
          ci1=dmax1(dabs(y(j)),dabs(yn(j)))*rto
          err=err+(temp3/(ato+ci1))**2
        end do
c -------- Atol and rtol are array.--------
                     else
        do j=1,neqn
          temp3=temp2*(y(j)-yjm2(j))
          y(j)=yjm1(j)+temp1*y(j)+temp3
          ci1=dabs(y(j))*rtol(j)
          err=err+(temp3/(atol(j)+ci1))**2
        end do
      end if
      err=sqrt(err/neqn)
      return 
      end
c ----------------------------------------------
c     End of subroutine rtstep.
c ---------------------------------------------- 
c      
      subroutine rkstep(neqn,npdes,t,h,y,ye,f,fd2,fa,fr,fw,yks,
     &yrk1,yrk2,yrk3,yrk4,yrk5,yrk0,yn,ytmp,y2,yerrA,yerrR,fnc,
     &errA,errR,errD2,atol,rtol,ntol,beta,frjac,ijac,iwork,idid,nell)	
c ----------------------------------------------
c  PIROCK: five stage IMEX partitioned method
c-----------------------------------------------
c             Declarations
c-----------------------------------------------
       double precision y(neqn),yn(*),ye(*),
     & atol(*),rtol(*),
     & t,h,temp1,temp2,temp3,ato,rto,
     & yks(neqn),beta,fnc(neqn),yrk0(neqn),ytmp(neqn),y2(neqn),
     & yrk1(neqn),yrk2(neqn),yrk3(neqn),yrk4(neqn),yrk5(neqn),
     & yerrA(neqn),yerrR(neqn),frjac(*),
     & errA,errR,errD2,alpha,gamma,alpha2,alpha3,
     & alphainv2,cbeta,h2,h3,h6,h23,h34,
     & b1,b2,b3,bb1,bb2,cbeta2,tmp
       integer neqn,npdes,i,j,ijac(*),im,ix,iwork(*),idid,nell
       external f,fd2,fa,fr,fw
			 logical is_frjac
c coefficients of PRK method
       gamma=1.d0-sqrt(0.5d0)
			 alpha=1.d0-2.d0*gamma
			 alphainv2=h*0.5d0/alpha
			 alpha=alpha*h
			 cbeta=h-2.d0*beta
			 if (nell.eq.1) cbeta=0.d0
			 cbeta2=2.d0*cbeta/3.d0
			 h2=0.5d0*h
			 h3=h/3.d0
			 h4=0.25d0*h
			 h6=0.5d0*h3
			 h23=2*h3
			 h34=3*h4
c embedded method for F_A
			 b2=0.3d0*h
			 b3=-0.5d0*b2
			 b1=-b2-b3
c embedded method for F_D2
       bb1=0.5d0*h34
			 bb2=-bb1
c
       alpha3=h23*gamma 
			 gamma=gamma*h
       alpha2=h23-gamma
			
c initialization	
			 do i=1,neqn
			 yrk1(i)=yks(i)
			 yrk2(i)=yks(i)
			 yrk3(i)=yks(i)
			 yrk4(i)=yks(i)
			 yrk5(i)=yks(i)
			 yerrA(i)=0.d0
			 yerrR(i)=0.d0
		   end do
c -------- First stage.--------
       if (iwork(21).eq.1) then
       is_frjac=.true.
       call ieuler(neqn,npdes,fr,t,gamma,yks,yrk1,ytmp,
     &   fnc,frjac,ijac,is_frjac,iwork,idid)
       do i=1,neqn
			 yrk2(i)=yks(i)+alpha*fnc(i)
			 yrk3(i)=yks(i)+(alpha+gamma)*fnc(i)
			 yrk5(i)=yks(i)+alpha2*fnc(i)
			 ye(i)=ye(i)+h2*fnc(i)
			 yerrR(i)=-h6*fnc(i)
			 end do
			 end if
       call f(neqn,t+beta,yrk1,fnc)
			 do i=1,neqn
			 y2(i)=fnc(i)
			 if (nell.eq.0) then
			 yrk2(i)=yrk2(i)+cbeta*fnc(i)
			 yrk5(i)=yrk5(i)+cbeta2*fnc(i)
			 end if
			 end do
c Noise terms (stochastic Ito noise)
       if (iwork(22).eq.1) then
			 do i=1,neqn
			 yrk0(i)=yrk2(i)
			 end do
       if (iwork(21).eq.1) then
					 call dampreaction(yrk0,neqn, npdes, frjac, ijac)
		   end if
			 call fw(neqn,t,yrk0,fnc)
			 tmp=sqrt(h)
			 do i=1,neqn
			 ye(i)=ye(i)+tmp*fnc(i)
			 end do
			 end if
c ------------------------------
		   if (iwork(20).eq.1) then
			 call fa(neqn,t,yrk1,fnc)
			 do i=1,neqn
			 yrk2(i)=yrk2(i)+h*fnc(i)
			 yrk3(i)=yrk3(i)+alpha*fnc(i)
			 yrk4(i)=yrk4(i)+h3*fnc(i)
			 ye(i)=ye(i)+h4*fnc(i)
			 yerrA(i)=b1*fnc(i)
			 end do
			 end if
			
			 if (iwork(24).eq.1) then
			 call fd2(neqn,t,yrk1,fnc)
			 do i=1,neqn
			 yrk2(i)=yrk2(i)+h*fnc(i)
			 yrk3(i)=yrk3(i)+alpha*fnc(i)
c for embbeded method of fd2
       yks(i)=bb1*fnc(i)
			 ye(i)=ye(i)+h4*fnc(i)
c save fnc
       yrk1(i)=fnc(i)
			 end do
			 end if
			
c -------- Second stage.--------
     	 if (iwork(21).eq.1) then
       do i=1,neqn
			 yrk0(i)=yrk2(i)
			 end do
			 is_frjac=.false.
       call ieuler(neqn,npdes,fr,t+h-gamma,gamma,yrk0,yrk2,
     &   ytmp,fnc,frjac,ijac,is_frjac,iwork,idid)
			 iwork(17)=iwork(17)+neqn/npdes
       do i=1,neqn
			 yrk5(i)=yrk5(i)+alpha3*fnc(i)
			 ye(i)=ye(i)+h2*fnc(i)
			 yerrR(i)=yerrR(i)+h6*fnc(i)
			 end do
			 end if
c -------- Third stage.--------
       call f(neqn,t+beta,yrk3,fnc)
       do i=1,neqn
			 y2(i)=fnc(i)-y2(i)
			 end do
c damping of finite difference
			 if (iwork(21).eq.1) then
			 call dampreaction(y2,neqn, npdes, frjac, ijac)
			 if (nell.eq.0) call dampreaction(y2,neqn, npdes, frjac, ijac)
			 end if
			 do i=1,neqn
			 ye(i)=ye(i)+alphainv2*y2(i)
			 end do
c -------- Fourth stage.--------
     	 if (iwork(20).eq.1.or.iwork(24).eq.1) then
       if (iwork(20).eq.1) then
			 call fa(neqn,t+h3,yrk4,fnc)
			 do i=1,neqn
			 yerrA(i)=yerrA(i)+b2*fnc(i)
			 end do	
			 if (iwork(24).eq.1) then
			 do i=1,neqn
			 fnc(i)=fnc(i)+yrk1(i)
			 end do
			 end if
			 else
			 do i=1,neqn
			 fnc(i)=yrk1(i)
			 end do
			 end if
			 if (iwork(21).eq.1) 
     &    call dampreaction(fnc,neqn, npdes, frjac, ijac)
			 do i=1,neqn
			 yrk5(i)=yrk5(i)+h23*fnc(i)
			 end do	
		   end if
c -------- Firth stage.--------
     	 if (iwork(20).eq.1) then
       call fa(neqn,t+h23,yrk5,fnc)
			 do i=1,neqn
			 ye(i)=ye(i)+h34*fnc(i)
			 yerrA(i)=yerrA(i)+b3*fnc(i)
			 end do	
		   end if
			 if (iwork(24).eq.1) then
			 call fd2(neqn,t,yrk5,fnc)
			 do i=1,neqn
			 ye(i)=ye(i)+h34*fnc(i)
c for embbeded method of fd2
       yks(i)=yks(i)+bb2*fnc(i)
			 end do
			 end if
c compensated summation
       do i=1,neqn
			 tmp=y(i)
			 y(i)=y(i)+ye(i)
			 ye(i)=ye(i)+(tmp-y(i))
			 end do
c ----- Error estimator
				errA=0.0d0
				errR=0.0d0
				errD2=0.0d0
c damping of errR (Shampine's idea)
         if (iwork(21).eq.1)
     &       call dampreaction(yerrR,neqn, npdes, frjac, ijac)
c -------- Atol and rtol are scalar.--------
       if (ntol.eq.0) then
        ato=atol(1)
        rto=rtol(1)
        do j=1,neqn
          temp1=dmax1(dabs(y(j)),dabs(yn(j)))*rto
          if (iwork(20).eq.1) errA=errA+(yerrA(j)/(ato+temp1))**2
					if (iwork(21).eq.1) errR=errR+(yerrR(j)/(ato+temp1))**2
					if (iwork(24).eq.1) errD2=errD2+(yks(j)/(ato+temp1))**2
        end do
c -------- Atol and rtol are array.--------
       else
        do j=1,neqn
          temp1=dabs(y(j))*rtol(j)
          if (iwork(20).eq.1) errA=errA+(yerrA(j)/(atol(j)+temp1))**2
					if (iwork(21).eq.1) errR=errR+(yerrR(j)/(atol(j)+temp1))**2
					if (iwork(24).eq.1) errD2=errD2+(yks(j)/(atol(j)+temp1))**2
        end do
       end if
c (to take into account the order 3 of the FA method)
       if (iwork(20).eq.1) errA=(errA/neqn)**(0.3333333333333333d0)
			 if (iwork(21).eq.1) errR=sqrt(errR/neqn)
			 if (iwork(24).eq.1) errD2=sqrt(errD2/neqn)
       return
      end
c ----------------------------------------------
c     End of subroutine rkstep.
c ---------------------------------------------- 
      subroutine dampreaction(fnc,neqn, npdes, frjac, ijac)
			double precision fnc(neqn),frjac(*)
			integer neqn,npdes,i,ix,im,ijac(*)
			im=1
		  do ix=1,neqn,npdes
      CALL SOL (npdes, npdes, frjac(im), fnc(ix), ijac(ix))
      im=im+npdes**2
      end do
c-----------------------------------------------
c             Declarations
c-----------------------------------------------
      return 
			end
c ------------------------
      subroutine ieuler(neqn,npdes,fr,t,h,y0,y1,ytmp,fnc,
     &   frjac,ijac,is_frjac,iwork,idid)
c-----------------------------------------------
c             Declarations
c-----------------------------------------------
       double precision err,err2,t,h,
     & y0(neqn),y1(neqn),fnc(neqn),ytmp(neqn),frjac(*)
       integer neqn,i,j,k,ix,im,irec
			 external fr
			 integer ier,ijac(*),iwork(*),idid
			 logical is_frjac,dojac
       do i=1,neqn
			 fnc(i)=0.d0
			 end do
			 err=1.d0
c recompute jacobian if iter>irec
			 irec=5 
			 
			 im=1
			 do ix=1,neqn,npdes
			 do k=1,50
			 dojac=(is_frjac.and.k.eq.1).or.k.gt.irec
			 call fr(neqn,npdes,ix,t,y1(ix),fnc(ix),frjac(im),dojac)
			 iwork(17)=iwork(17)+1
			 if (dojac) iwork(18)=iwork(18)+1
		   do i=1,npdes
			 ytmp(ix+i-1)=y1(ix+i-1)-y0(ix+i-1)-h*fnc(ix+i-1)
			 end do
			 if (dojac) then
			 do i=1,npdes
			 do j=1,npdes
			 frjac(im+(i-1)+(j-1)*npdes)=-h*frjac(im+(i-1)+(j-1)*npdes)
			 end do
			 frjac(im+(i-1)*(npdes+1))=frjac(im+(i-1)*(npdes+1))+1.d0
			 end do
			 CALL DEC (npdes, npdes, frjac(im), ijac(ix), IER)
			 if (IER.NE.0) then
			 write (6,*) 'WARNING; BUG IN DEC TRIANGULATION'
c			 pause
			 end if
			 end if
       CALL SOL (npdes, npdes, frjac(im), ytmp(ix), ijac(ix))
			 err2=err
			 err=0.d0
			 do i=1,npdes
			 y1(ix+i-1)=y1(ix+i-1)-ytmp(ix+i-1)
			 err=err+dabs(ytmp(ix+i-1))
			 end do		
			 if (err.le.1.d-13) goto 17	
c			 if (err.eq.0.d0.or.(err.ge.err2.and.k.ge.3)) goto 17
c			 if (err.le.1.d-13.or.(err.ge.err2.and.k.ge.3)) goto 17
			 end do
			 write (6,*) 'WARNING; NEWTON ITERATION FAILED TO CONVERGE',
     &   ix,err,err2,k,h,y0(ix),y0(ix+1),y1(ix),y1(ix+1)
c		   pause
   17  continue
	     im=im+npdes**2
       iwork(13)=max(k,iwork(13))
       end do			 
			end
c ---------------------------------------------------------------
c     compute fd,fd2,fa together (non-stiff vector field case)
c
      subroutine rkstep0(neqn,npdes,t,h,y,ye,f,fd2,fa,fr,fw,fn,
     &yrk1,yrk2,yrk3,yrk4,yrk5,yrk0,yn,ytmp,y2,yerrA,yerrR,fnc,
     &errD,errD2,errA,errR,atol,rtol,ntol,frjac,ijac,iwork,idid,nell)	
c ----------------------------------------------
c  PIROCK: five stage IMEX partitioned method
c-----------------------------------------------
c             Declarations
c-----------------------------------------------
       double precision y(neqn),yn(*),ye(*),
     & atol(*),rtol(*),
     & t,h,temp1,temp2,temp3,ato,rto,
     & fn(neqn),fnc(neqn),yrk0(neqn),ytmp(neqn),y2(neqn),
     & yrk1(neqn),yrk2(neqn),yrk3(neqn),yrk4(neqn),yrk5(neqn),
     & yerrA(neqn),yerrR(neqn),frjac(*),
     & errD,errD2,errA,errR,alpha,gamma,alpha2,alpha3,
     & alphainv2,h2,h3,h6,h23,h34,
     & b1,b2,b3,tmp
       integer neqn,npdes,i,j,ijac(*),im,ix,iwork(*),idid,nell
       external f,fd2,fa,fr,fw
			 logical is_frjac
c coefficients of PRK method
       gamma=1.d0-sqrt(0.5d0)
			 alpha=1.d0-2.d0*gamma
			 alphainv2=h*0.5d0/alpha
			 alpha=alpha*h
			 h2=0.5d0*h
			 h3=h/3.d0
			 h4=0.25d0*h
			 h6=0.5d0*h3
			 h23=2*h3
			 h34=3*h4
c embedded method for F_A
			 b2=0.3d0*h
			 b3=-0.5d0*b2
			 b1=-b2-b3
       alpha3=h23*gamma 
			 gamma=gamma*h
       alpha2=h23-gamma
			
			
c initialization	
			 do i=1,neqn
			 yrk1(i)=yn(i)
			 yrk2(i)=yn(i)
			 yrk3(i)=yn(i)
			 yrk4(i)=yn(i)
			 yrk5(i)=yn(i)
			 yerrA(i)=0.d0
			 yerrR(i)=0.d0
		   end do
c -------- First stage.--------
       if (iwork(21).eq.1) then
       is_frjac=.true.
       call ieuler(neqn,npdes,fr,t,gamma,yn,yrk1,ytmp,
     &   fnc,frjac,ijac,is_frjac,iwork,idid)
       do i=1,neqn
			 yrk2(i)=yn(i)+alpha*fnc(i)
			 yrk3(i)=yn(i)+(alpha+gamma)*fnc(i)
			 yrk5(i)=yn(i)+alpha2*fnc(i)
			 ye(i)=ye(i)+h2*fnc(i)
			 yerrR(i)=-h6*fnc(i)
			 end do
			 end if
c Noise terms (stochastic Ito noise)
       if (iwork(22).eq.1) then
			 do i=1,neqn
			 yrk0(i)=yrk2(i)
			 end do
       if (iwork(21).eq.1) then
					 call dampreaction(yrk0,neqn, npdes, frjac, ijac)
		   end if
			 call fw(neqn,t,yrk0,fnc)
			 tmp=sqrt(h)
			 do i=1,neqn
			 ye(i)=ye(i)+tmp*fnc(i)
			 end do
			 end if
c ------------------------------
			 if (iwork(20).eq.1) then
			 call fa(neqn,t,yrk1,fnc)
			 do i=1,neqn
			 fnc(i)=fnc(i)+fn(i)
			 end do
			 else
			 do i=1,neqn
			 fnc(i)=fn(i)
			 end do
			 end if
			 if (iwork(24).eq.1) then
			 call fd2(neqn,t,yrk1,ytmp)
			 do i=1,neqn
			 fnc(i)=fnc(i)+ytmp(i)
			 end do
			 end if
			 do i=1,neqn
			 yrk2(i)=yrk2(i)+h*fnc(i)
			 yrk3(i)=yrk3(i)+alpha*fnc(i)
			 yrk4(i)=yrk4(i)+h3*fnc(i)
			 ye(i)=ye(i)+h4*fnc(i)
			 yerrA(i)=b1*fnc(i)
			 end do
c -------- Second stage.--------
     	 if (iwork(21).eq.1) then
       do i=1,neqn
			 yrk0(i)=yrk2(i)
			 end do
			 is_frjac=.false.
       call ieuler(neqn,npdes,fr,t+h-gamma,gamma,yrk0,yrk2,
     &   ytmp,fnc,frjac,ijac,is_frjac,iwork,idid)
			 iwork(17)=iwork(17)+neqn/npdes
       do i=1,neqn
			 yrk5(i)=yrk5(i)+alpha3*fnc(i)
			 ye(i)=ye(i)+h2*fnc(i)
			 yerrR(i)=yerrR(i)+h6*fnc(i)
			 end do
			 end if
c -------- Fourth stage.--------
       call f(neqn,t+h3,yrk4,fnc)
			 if (iwork(20).eq.1) then
			 call fa(neqn,t+h3,yrk4,ytmp)
			 do i=1,neqn
			 fnc(i)=fnc(i)+ytmp(i)
			 end do
			 end if
			 if (iwork(24).eq.1) then
			 call fd2(neqn,t+h3,yrk4,ytmp)
			 do i=1,neqn
			 fnc(i)=fnc(i)+ytmp(i)
			 end do
			 end if
			 do i=1,neqn
			 yerrA(i)=yerrA(i)+b2*fnc(i)
			 end do	
			 if (iwork(21).eq.1) 
     &    call dampreaction(fnc,neqn, npdes, frjac, ijac)
			 do i=1,neqn
			 yrk5(i)=yrk5(i)+h23*fnc(i)
			 end do	
c -------- Firth stage.--------
       call f(neqn,t+h23,yrk5,fnc)
			 if (iwork(20).eq.1) then
			 call fa(neqn,t+h23,yrk5,ytmp)
			 do i=1,neqn
			 fnc(i)=fnc(i)+ytmp(i)
			 end do
			 end if
			 if (iwork(24).eq.1) then
			 call fd2(neqn,t+h23,yrk5,ytmp)
			 do i=1,neqn
			 fnc(i)=fnc(i)+ytmp(i)
			 end do
			 end if
			 do i=1,neqn
			 ye(i)=ye(i)+h34*fnc(i)
			 yerrA(i)=yerrA(i)+b3*fnc(i)
			 end do	
c compensated summation
       do i=1,neqn
			 tmp=yn(i)
			 y(i)=yn(i)+ye(i)
			 ye(i)=ye(i)+(tmp-y(i))
			 end do
c ----- Error estimator
				errA=0.0d0
				errR=0.0d0
c damping of errR (Shampine's idea)
         if (iwork(21).eq.1)
     &       call dampreaction(yerrR,neqn, npdes, frjac, ijac)
c -------- Atol and rtol are scalar.--------
       if (ntol.eq.0) then
        ato=atol(1)
        rto=rtol(1)
        do j=1,neqn
          temp1=dmax1(dabs(y(j)),dabs(yn(j)))*rto
          errA=errA+(yerrA(j)/(ato+temp1))**2
					if (iwork(21).eq.1) errR=errR+(yerrR(j)/(ato+temp1))**2
        end do
c -------- Atol and rtol are array.--------
       else
        do j=1,neqn
          temp1=dabs(y(j))*rtol(j)
          errA=errA+(yerrA(j)/(atol(j)+temp1))**2
					if (iwork(21).eq.1) errR=errR+(yerrR(j)/(atol(j)+temp1))**2
        end do
       end if
c (to take into account the order 3 of the FA method)
       errA=(errA/neqn)**(0.3333333333333333d0)
			 if (iwork(21).eq.1) errR=sqrt(errR/neqn)
			 errD=errA
			 errD2=errA
       return
      end
c ----------------------------------------------
c     End of subroutine rkstep.
c ---------------------------------------------- 
  

      subroutine mdegr(mdeg,mp,ms)
c-------------------------------------------------------------          
c       Find the optimal degree.
c       MP(1): pointer which select the degree in ms(i)\1,2,..
c             such that mdeg<=ms(i).
c       MP(2): pointer which gives the corresponding position
c       of a_1 in the data recf for the selected degree.
c-------------------------------------------------------------        
c ---------------------------------------------- 
c             Declarations
c ---------------------------------------------- 
      integer ms(46),mp(2),mdeg,i
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c             Initialisations
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
      mp(2)=1
c -------- Find the degree.--------     
      do i=1,46
        if ((ms(i)/mdeg).ge.1) then
          mdeg=ms(i)
          mp(1)=i
          return
        end if
      mp(2)=mp(2)+ms(i)*2-1
      end do
      return
      end  
c ----------------------------------------------
c     End of subroutine mdegr.
c ----------------------------------------------
c   
      subroutine rocktrho(neqn,t,y,f,yn,fn,work,z,fz,eigmax,
     &                   uround,idid,iwork)
c------------------------------------------------------------ 
c     Rocktrho compute eigmax, a close upper bound of the
c     spectral radius of the Jacobian matrix using a 
c     power method (J.N. Franklin (matrix theory)). 
c     The algorithm used is a small change (initial vector
c     and stopping criteria) of that of
c     Sommeijer-Shampine-Verwer, implemented in RKC.
c-------------------------------------------------------------
c             Declarations
c-------------------------------------------------------------
       double precision y(neqn),yn(neqn),fn(neqn),z(neqn),
     & work(*),fz(neqn),t,eigmax,eigmaxo,sqrtu,uround,znor,
     & ynor,quot,dzyn,dfzfn,safe
       integer iwork(12),neqn,n5,i,iter,maxiter,nind,ntest,
     & ind,idid  
       parameter (maxiter=50)
       parameter (safe=1.2d0)
       external f
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
c             Initialisations
c *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***     
      sqrtu=sqrt(uround)
      ynor=0.d0
      znor=0.d0
      n5=4*neqn
c ------ The initial vectors for the power method are yn --------
c       and yn+c*f(v_n), where vn=f(yn) a perturbation of yn 
c       (if iwork(6)=0) or a perturbation of the last computed
c       eigenvector (if iwork(6).neq.0). 
c
      if (iwork(6).eq.0) then
        do i=1,neqn
          fz(i)=fn(i)
        end do
        call f(neqn,t,fz,z)
        iwork(9)=iwork(9)+1
                         else
        do i=1,neqn
          z(i)=work(n5+i)
        end do
      end if
c ------ Perturbation.--------
      do i=1,neqn
        ynor=ynor+yn(i)**2
        znor=znor+z(i)**2
      end do
      ynor=sqrt(ynor)
      znor=sqrt(znor)
c ------ Normalization of the vector z so that --------
c        the difference z-yn lie in a circle 
c        around yen (ice has a constant modules).
c
      if (ynor.ne.0.d0.and.znor.ne.0.d0) then
        dzyn=ynor*sqrtu
        quot=dzyn/znor
        do i=1,neqn
          z(i)=yn(i)+z(i)*quot
        end do
      elseif(ynor.ne.0.d0) then
        dzyn=ynor*sqrtu
        do i=1,neqn
          z(i)=yn(i)+yn(i)*sqrtu
        end do
      elseif(znor.ne.0.d0) then
          dzyn=uround
          quot=dzyn/znor
          do i=1,neqn
            z(i)=z(i)*quot
          end do
      else
        dzyn=uround
        do i=1,neqn
          z(i)=dzyn
        end do
      end if
c ------ Start the power method.--------
      eigmax=0.d0
      do iter=1,maxiter
        call f(neqn,t,z,fz)
        iwork(9)=iwork(9)+1
        dfzfn=0.d0
        do i=1,neqn
          dfzfn=dfzfn+(fz(i)-fn(i))**2
        end do
        dfzfn=dsqrt(dfzfn)
        eigmaxo=eigmax
        eigmax=dfzfn/dzyn
        eigmax=safe*eigmax
c ------ The stopping criteria is based on a 
c        relative error between two successive
c        estimation ``eigmax'' of the spectral 
c        radius.
c 
        if (iter.ge.2.and.dabs(eigmax-eigmaxo)
     &    .le.(eigmax*0.05d0)) then
c ----- The last eigenvector is stored.--------
          do i=1,neqn
            work(n5+i)=z(i)-yn(i)
          end do
          return
        end if
c ----- The next z is defined by --------
c       z_new=yn+coef*(fz-fn) where
c       coef is chosen so that
c       norm(z_new-yn)=norm(z_old-yn).
c
        if (dfzfn.ne.0.d0) then
          quot=dzyn/dfzfn
          do i=1,neqn
            z(i)=yn(i)+(fz(i)-fn(i))*quot
          end do
        else
c ----- The new z is defined by an arbitrary --------
c       perturbation of the current approximation
c       of the eigenvector.
c
          nind=neqn
          ntest=0
          ind=1+mod(iter,nind)
          if (z(ind).ne.yn(ind).or.ntest.eq.10) then
            z(ind)=yn(ind)-(z(ind)-yn(ind))
          else 
            nind=neqn+ind
            ntest=ntest+1
          end if
        end if
      end do
      write(6,*) 'convergence failure in the 
     & spectral radius computation'
      idid=-3
      return
      end
      
         
        
        
        
           
         
         
       
                        
                  
       
       
           
         
      
      
     


