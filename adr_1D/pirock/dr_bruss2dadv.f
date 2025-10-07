c * * * * * * * * * * * * * * * * * * * * * * * * *
c    Driver for PIROCK
c * * * * * * * * * * * * * * * * * * * * * * * * *
c ----- to integrate with pirock.f ----- 
      include 'pirock.f' 
      include 'decsol.f'  
c --- Problem driver and dimension parameters
			include 'pb_bruss2dadv.f' 
      parameter (nsd=400,npdes=0,neqn=nsd*nsd*2)
c ----------------------------------------------------
			implicit double precision (a-h,o-z)
      external fd,fd2,fa,fr,fw
c ----- to integrate with pirock.f 
      dimension y(neqn),work(15*neqn),frjac(neqn*npdes) 
      integer iwork(25),idid,ijac(neqn)
c ----- required tolerance -----
			 atol=1.d-3
			 rtol=atol
c ----- initial step size -----
			h=1.d-5
			write (6,*) 'initial step size h=',h
c--------------------------------------------------------
c     Initialize iwork: 
c      iwork(1)=1  RHODIFF returns an upper bound for the spectral radius.
c      iwork(2)=1  The Jacobian of diffusion is constant (RHODIFF is called once).
c      iwork(3)=0  Return and solution at tend.
c      iwork(4)=0  Atol and rtol are scalars.
      iwork(1)=1
      iwork(2)=1
      iwork(3)=0
      iwork(4)=0
c--------------------------------------------------------
c     iwork(19)   =2 Stepsize control with  memory
c                 =1 Stepsize control without memory
c                 =0 Constant stepsize
c     iwork(20)   =1 Enable F_A (advection or nonstiff reaction)
c     iwork(21)   =1 Enable F_R (stiff reaction)
c     iwork(22)   =1 Enable F_W (noise, constant stepsize)
c     iwork(23)   =1 Verbose (print stepsizes and errors)
c     iwork(24)   =0 (symmetric diffusion operator)
c--------------------------------------------------------
			iwork(19)=2
			iwork(23)=0
			iwork(24)=0
c
			iwork(20)=1
			iwork(21)=0
			iwork(22)=0
	
c iwork for stats
      do i=5,18
			iwork(i)=0
			end do
			
      call init(nsd,t,tend,y)

c ----- integration -----
			write (6,*) 'tol',atol 
			CALL CPU_TIME(time0)
c ----- to integrate with rock2.f   
      call pirock(neqn,npdes,t,tend,h,y,fd,fd2,fa,fr,fw,atol,rtol,
     &           frjac,ijac,work,iwork,idid)  
      CALL CPU_TIME(time1)
			write (6,*) 'CPU time',time1-time0
c ----- print statistics -----
      write(6,*) 'The value of IDID is',idid
      write(6,*) 'Max estimation of the spectral radius=',iwork(11)
      write(6,*) 'Min estimation of the spectral radius=',iwork(12)
      write(6,*) 'Max spectral radius (advection)=',iwork(14)
      write(6,*) 'Min spectral radius (advection)=',iwork(15)
      write(6,*) 'Max number of stages used=',iwork(10)
      write(6,*) 'Number of f eval. for the spectr. radius=',iwork(9)
			write(6,*) 'Max number of iterations used=',iwork(13)
      write (6,91) iwork(5),iwork(16),iwork(6),
     &   iwork(7),iwork(8),iwork(13)
 91   format(' Number of f evaluations=',i7,' fA evaluations=',i7,
     &      ' steps=',i7,' accpt=',i7,' rejct=',i7,' max iter',i4) 
		 
		  write (6,*) 'Number of reaction VF',
     &   iwork(17),(iwork(17)*npdes)/neqn
		  write (6,*) 'Number of reaction Jacobian',
     &   iwork(18),(iwork(18)*npdes)/neqn
		  
			call solout(neqn,t,tend,y,work)

c--------------------------------------------------------
c     End of main program
c--------------------------------------------------------
      end      


