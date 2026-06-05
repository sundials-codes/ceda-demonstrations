c * * * * * * * * * * * * * * * * * * * * * * * * *
c    Driver for PIROCK
c * * * * * * * * * * * * * * * * * * * * * * * * *
c ----- to integrate with pirock.f -----
      include 'pirock.f'
      include 'decsol.f'
c --- Problem driver and dimension parameters
	include 'pb_bruss2dreac.f'
      parameter (nsd=200,npdes=2,neqn=nsd*nsd*npdes)
c ----------------------------------------------------
	implicit double precision (a-h,o-z)
      external fd,fd2,fa,fr,fw
c --- common parameters for the problem -----
      common/trans/atol,rtol,alf,amult,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth,
     &    iout,nout
c ----- to integrate with pirock.f
      dimension y(neqn),work(15*neqn),frjac(neqn*npdes)
      integer*8 iwork(25)
      integer idid,ijac(neqn)
      logical fixedstep
c --- namelist definition
      namelist /inputs/ alf,brussa,brussb,atol,rtol,h,tend,nout
c --- read input from namelist file (if it exists) ---
      open(10, file='rd_2D_pirock_params.txt', status='old', err=100)
      read(10, nml=inputs)
      close(10)
      goto 110

  100 continue
      write(6,*) 'Could not open namelist file'
c ----- initial step size -----
  110	if (h .le. 0.d0) then
          fixedstep=.false.
          h=1.d-3
          write(6,*) 'Initial step size h=',h
      else
          fixedstep=.true.
          write(6,*) 'Fixed step size h=',h
      end if
c note that we multiply by input tolerances and final time by 1.d0 since Python doesn't write values with '.d'
      atol=atol*1.d0
      rtol=rtol*1.d0
      tend=tend*1.d0

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
      if (fixedstep) then
          iwork(19)=0
      else
          iwork(19)=2
      end if
	iwork(23)=0
	iwork(24)=0
c
	iwork(20)=0
	iwork(21)=1
	iwork(22)=0

c iwork for stats
      do i=5,18
	    iwork(i)=0
	end do

c note that we define the final time from the namelist input file, and not the hard-coded value in init()
      call init(nsd,t,tend2,y)
      iout = 0
      call solout(neqn,t,tend,y,work)

c ----- integration -----
	write (6,*) 'rtol',rtol
	write (6,*) 'atol',atol
      write (6,*) 'nout',nout
c ----- to integrate with pirock.f
      time_tot = 0.d0
      dtout = tend/nout
      do i=1,nout
        tout = i*dtout
        iout = i
        CALL CPU_TIME(time0)
        call pirock(neqn,npdes,t,tout,h,y,fd,fd2,fa,fr,fw,atol,rtol,
     &              frjac,ijac,work,iwork,idid)
        t = tout
        call solout(neqn,t,tend,y,work)
        CALL CPU_TIME(time1)
        time_tot = time_tot + time1 - time0
      end do

c ----- print statistics -----
	write(6,*) 'CPU time',time_tot
      write(6,*) 'The value of IDID is',idid
      write(6,*) 'Max estimation of the spectral radius=',iwork(11)
      write(6,*) 'Min estimation of the spectral radius=',iwork(12)
      write(6,*) 'Max spectral radius (advection)=',iwork(14)
      write(6,*) 'Min spectral radius (advection)=',iwork(15)
      write(6,*) 'Max number of stages used=',iwork(10)
      write(6,*) 'Number of f eval. for the spectr. radius=',iwork(9)
	write(6,*) 'Max number of iterations used=',iwork(13)
      write(6,*) 'Number of f evaluations=',iwork(5),' fA evaluations=',
     &   iwork(16),' steps=',iwork(6),' accpt=',iwork(7),' rejct=',
     &   iwork(8),' max iter',iwork(13)

	write (6,*) 'Number of reaction VF',
     &   iwork(17),(iwork(17)*npdes)/neqn
	write (6,*) 'Number of reaction Jacobian',
     &   iwork(18),(iwork(18)*npdes)/neqn

c--------------------------------------------------------
c     End of main program
c--------------------------------------------------------
      end


