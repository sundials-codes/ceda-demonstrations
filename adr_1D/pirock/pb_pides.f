c--------------------------------------------------------
c    PIDES - integro-differential problem
c--------------------------------------------------------
      SUBROUTINE init(nsd,t,tend,y)
			implicit double precision (a-h,o-z)
      double precision  y(*)
			
c --- common parameters for the problem -----
      common/trans/diff,fac,ns,nssq,ainvns,imeth

c ----- dimensions -----
			neqn=nsd
		  ns=nsd
			nssq=ns**2
			ainvns=1.d0/ns
			api=4.d0*atan(1.d0)
			diff=1.d0
			fac=1.d-2
			

      write(6,*) 'Integration of the '
     &   ,'integro-differential problem, ns=',ns
c ----- initial and end point of integration -----
      t=0.0d0
      tend=1.d0
c ----- initial values -----
      do i=1,ns
        xx=i*ainvns
        y(i)=0.5d0*(cos(api*xx)+1.d0)
      end do
			
      return
      end
			
			SUBROUTINE solout(neqn,t,tend,y,ytmp)
			implicit double precision (a-h,o-z)
      double precision  y(neqn),ytmp(neqn),K0
			
c --- common parameters for the problem -----
      common/trans/diff,fac,ns,nssq,ainvns,imeth

c ----- file for solution -----
       open(8,file='sol.dat')
       rewind 8
		   write (8,*) (y(i),i=1,ns)
			 write(6,*) 'Solution is tabulated in file sol.dat'
			 close(8)
      return
      end


c--------------------------------------------------------
c     The subroutine RHO gives an estimation of the spectral 
c     radius of the Jacobian matrix of the diffusion. This
c     is a bound for the whole interval and thus RHO is called
c     once.
c--------------------------------------------------------
      double precision function rhodiff(neqn,t,y)
      implicit double precision (a-h,o-z)
      common/trans/diff,fac,ns,nssq,ainvns,imeth
        rhodiff = 4.0d0*nssq*diff 
      return
      end 
c--------------------------------------------------------
c     The subroutine RHOADV gives an estimation of the spectral 
c     radius of the Jacobian matrix of the advection. This
c     is a bound for the whole interval and thus RHO is called
c     once.
c--------------------------------------------------------
      double precision function rhoadv(neqn,t,y)
      implicit double precision (a-h,o-z)
      common/trans/diff,fac,ns,nssq,ainvns,imeth
        rhoadv = 0.d0 
      return
      end 
c--------------------------------------------------------
c     The subroutine FBRUS compute the value of f(x,y) and
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fd(neqn,x,y,f)
c ----- brusselator with diffusion in 2 dim. space -----
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/diff,fac,ns,nssq,ainvns,imeth
      u0=0.5d0*(2.d0-sqrt(x))
      f(1)=nssq*diff*(u0-2.d0*y(1)+y(2))
			do i=2,neqn-1
			f(i)=nssq*diff*(y(i-1)-2.d0*y(i)+y(i+1))
			end do
			f(neqn)=nssq*diff*2.d0*(y(neqn-1)-y(neqn))
      return
      end  

c--------------------------------------------------------
c     The subroutine FA (costly reaction terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fa(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/diff,fac,ns,nssq,ainvns,imeth
      
			u0=0.5d0*(2.d0-sqrt(x))
			u04=u0**4
			
      do i=1,neqn
			f(i)=0.d0
			xx=i*ainvns
      do j=1,neqn-1
			yy=j*ainvns
      f(i) = f(i)-y(j)**4/(1+abs(xx-yy))**2
      end do
      f(i) = f(i)-( u04/(1+xx)**2	+ y(neqn)**4/(2.d0-xx)**2 )*0.5d0
      f(i) = f(i)*fac*ainvns
			end do
			
      return
      end  
c
c--------------------------------------------------------
c     The subroutine FR (reaction terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fr(neqn,npdes,ieqn,x,y,f,frjac,is_frjac)
      implicit double precision (a-h,o-z)
      write (6,*) 'WARNING DUMMY FUNCTION FR CALLED'
		  return
      end  
c--------------------------------------------------------
c     The subroutine FW (noise terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fw(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      write (6,*) 'WARNING DUMMY FUNCTION FR CALLED'
		  return
      end  
c--------------------------------------------------------
c     The subroutine FD2 (non-symmetric diffusion)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fd2(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
		  write (6,*) 'warning, dummy function fd2 called !!'
			do i=1,neqn
			f(i)=0.0d0
			end do
			return
      end