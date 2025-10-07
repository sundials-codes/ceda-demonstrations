c--------------------------------------------------------
c    Combustion problem
c--------------------------------------------------------

      SUBROUTINE init(nsd,t,tend,w)
			implicit double precision (a-h,o-z)
      double precision  w(*),k0,z0
			
c --- common parameters for the problem -----
			common /params/   K0,Z0,ns,nssq,is_rec
			common /pard1/   d1max
		  ns=nsd
      write(6,*) 'Integration of the '
     &   ,'combustion problem, ns=',ns

c problem parameters
      K0 = 0.005d0
      Z0 = 1.0d0
			Z0 = 1.d1
			
			write (6,*) 'K0',K0,'Z0',Z0
c ----- initial and end point of integration -----
      t=0.0d0
      tend=3.d0
			
			m = ns
			nssq=ns**2
      h = 1.d0/m
      do i = 1, m
       do j = 1, m
c        w((i-1)*m+j) = 1.0d-5
c        w(m**2+(i-1)*m+j) = 0.05623413251903d0
        w(((i-1)*m+j)*2-1) = 1.0d-5
        w(((i-1)*m+j)*2) = 0.05623413251903d0
       end do
      end do
			
			d1max=1.d-4
      return
      end
			
      SUBROUTINE soloutold(neqn,t,tend,y)
			implicit double precision (a-h,o-z)
      double precision  y(neqn),k0,z0
			
c --- common parameters for the problem -----
      common /params/   K0,Z0,ns,nssq,is_rec

c ----- file for solution -----
       open(8,file='sol.dat')
       rewind 8
		  write (8,*) t,((y(((j-1)*ns+i)*2-1),i=1,ns),j=1,ns),
     &     ((y(((j-1)*ns+i)*2),i=1,ns),j=1,ns)
			 close(8)
			 write(6,*) 'Solution is tabulated in file sol.dat'
      return
      end
			
			      SUBROUTINE solout(neqn,t,tend,y,ytmp)
			implicit double precision (a-h,o-z)
      double precision  y(neqn),ytmp(neqn),K0
			
c --- common parameters for the problem -----
      common /params/   K0,Z0,ns,nssq,is_rec

c ----- file for solution -----
       open(8,file='sol.dat')
       rewind 8
		  write (8,*) t,((y(((j-1)*ns+i)*2-1),i=1,ns),j=1,ns),
     &     ((y(((j-1)*ns+i)*2),i=1,ns),j=1,ns)

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
			double precision k0,z0,s0,s1
			common /params/   K0,Z0,ns,nssq,is_rec
			common /pard1/   d1max
			m=ns
			h = 1.0d0/m
      Z = Z0
			
c      s0 = 6d3*Z**3 + 8.0d0/(Z**3*h**2)
      s0 =  8.0d0/(Z**3*h**2)
      s1 = 6d3 + 8.0d0/h**2
      rhodiff = amax1(s0,s1)
			
c			write (6,*) 'd1max',d1max
c      rhodiff = 6d3 + min(1.2d0*d1max,1.d0) *8.0d0/h**2
c      write (6,*) 'rhodiff',rhodiff,d1max,8.0d0/h**2
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
        rhoadv = 0.d0 
      return
      end 
c--------------------------------------------------------
c     The subroutine FD compute the value of f(x,y) and
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fd(neqn,t,w,fw)
c ----- brusselator with diffusion in 2 dim. space -----
      implicit double precision (a-h,o-z)
			double precision k0,z0
			common /params/   K0,Z0,ns,nssq,is_rec
			common /pard1/   d1max
      double precision  t,w(neqn),fw(neqn)
      parameter         (m = 100)
      double precision  u(0:m+1,0:m+1),fu(m,m)
      double precision  v(0:m+1,0:m+1),fv(m,m)
      double precision  D1(0:m+1,0:m+1),D2(0:m+1,0:m+1)
      double precision  D1x(0:m,m),D1y(m,0:m),D2x(0:m,m),D2y(m,0:m)
      double precision  K,Z,sigma,C,E,E1,E2,c0,c1
      character*80 Comment
c
      d1max=0.d0
c			
      if (.not.(m**2 .eq. neqn/2)) print*,'NOTE: m**2 .neq. neqn/2',
     &    m,neqn
			ns=m
      h = 1.d0/m
      K = K0
      do i = 1, m
       do j = 1, m
c        u(i,j) = w((i-1)*m+j) 
c        v(i,j) = w(m**2+(i-1)*m+j) 
        u(i,j) = w(((i-1)*m+j)*2-1) 
        v(i,j) = w(((i-1)*m+j)*2) 
       end do
      end do
      Comment = 'Set boundary conds.'
      do i = 1,m
        u(i,0) = u(i,1)
        u(i,m+1) = u(i,m)
        v(i,0) = v(i,1)
        v(i,m+1) = v(i,m)
      end do
      do j = 0,m+1
        v(0,j) = v(1,j)
c        sigma = 1.0/v(1,j)**3
c        c0 = 1.d0/8.d0 - ns/(6.d0*sigma)
c        c1 = 1.d0/8.d0 + ns/(6.d0*sigma)
        sigma = v(1,j)**3
        c0 = 1.d0/8.d0 - ns*sigma/(6.d0)
        c1 = 1.d0/8.d0 + ns*sigma/(6.d0)
        u(0,j) = (1.d0 - c0*u(1,j))/c1
        v(m+1,j) = v(m,j)
c        sigma = 1.d0/v(m,j)**3
c        c0 = 1.d0/8.d0 - ns/(6.d0*sigma)
c        c1 = 1.d0/8.d0 + ns/(6.d0*sigma)
        sigma = v(m,j)**3
        c0 = 1.d0/8.d0 - ns*sigma/(6.d0)
        c1 = 1.d0/8.d0 + ns*sigma/(6.d0)
        u(m+1,j) = -c0*u(m,j)/c1
      end do
      Comment = 'Set diffusion coefficients'
      do i = 0, m+1
       do j = 0, m+1
        x = (i-0.5d0)*h
        y = (j-0.5d0)*h
        Z = 1.d0
        if (dabs(x-0.5d0).le.1.d0/6d0 .and. dabs(y-0.5d0).le.1.d0/6.d0)
     &Z = Z0
c        sigma = Z**3/v(i,j)**3
        sigma = Z**3
        E = u(i,j)
        if (1.le.i .and. i.le.m) then
          E1 = 0.5d0*ns*(u(i+1,j)-u(i-1,j))
         else if (i.eq.0) then
          E1 = ns*(u(i+1,j)-u(i,j))
         else if (i.eq.m+1) then
          E1 = ns*(u(i,j)-u(i-1,j))
        end if
       if (1.le.j .and. j.le.m) then
          E2 = 0.5d0*ns*(u(i,j+1)-u(i,j-1))
         else if (j.eq.0) then
          E2 = ns*(u(i,j+1)-u(i,j))
         else if (j.eq.m+1) then
          E2 = ns*(u(i,j)-u(i,j-1))
        end if
        C = dsqrt((E1**2+E2**2)/E**2)
c        D1(i,j) = 1.d0/((3.d0*sigma) + C)
        D1(i,j) = v(i,j)**3/((3.d0*sigma) + C*v(i,j)**3)
				d1max=max(d1max,D1(i,j))
				 if (v(i,j).le.0.d0) then
				 write (6,*) 'Error in fd, v(i,j) is negative!',v(i,j),i,j
				 end if
c        D2(i,j) = K*abs(v(i,j))**(2.5)
        D2(i,j) = K*abs(v(i,j))**(2.5)
				d1max=max(d1max,D2(i,j))
       end do
      end do
      Comment = 'Set x-coefficients'
      do i = 0, m
       do j = 1, m
        D1x(i,j) = (D1(i,j)+D1(i+1,j))/2.d0
        D2x(i,j) = (D2(i,j)+D2(i+1,j))/2.d0
       end do
      end do
      Comment = 'Set y-coefficients'
      do i = 1, m
       do j = 0, m
        D1y(i,j) = (D1(i,j)+D1(i,j+1))/2.d0
        D2y(i,j) = (D2(i,j)+D2(i,j+1))/2.d0
       end do
      end do
			
			
      Comment = 'Compute F-functions'
      do i = 1, m
       do j = 1, m
        fu(i,j) = 
     &  + nssq*(D1x(i-1,j)*(u(i-1,j)-u(i,j)) 
     &     - D1x(i,j)*(u(i,j)-u(i+1,j)))
     &  + nssq*(D1y(i,j-1)*(u(i,j-1)-u(i,j)) 
     &     - D1y(i,j)*(u(i,j)-u(i,j+1)))
        fv(i,j) = 
     &  + nssq*(D2x(i-1,j)*(v(i-1,j)-v(i,j)) 
     &     - D2x(i,j)*(v(i,j)-v(i+1,j)))
     &  + nssq*(D2y(i,j-1)*(v(i,j-1)-v(i,j)) 
     &     - D2y(i,j)*(v(i,j)-v(i,j+1)))
       end do
      end do
			
c reaction
       if (is_rec.eq.1) then
       do i = 1, m
       do j = 1, m
			  x = (i-0.5d0)*h
        y = (j-0.5d0)*h
        Z = 1.d0
        if (dabs(x-0.5d0).le.1.d0/6.d0 .and. dabs(y-0.5d0).le.1.d0/6.d0) 
     &   then
				 Z=Z0
			  end if		
        sigma = Z**3/v(i,j)**3
			  fu(i,j) = fu(i,j)+sigma*(v(i,j)**4 - u(i,j))
        fv(i,j) = fv(i,j)-sigma*(v(i,j)**4 - u(i,j))
       end do
      end do
			end if
			
      do i = 1, m
       do j = 1, m
c        fw((i-1)*m+j) = fu(i,j)
c        fw(m**2+(i-1)*m+j) = fv(i,j)		
        fw(((i-1)*m+j)*2-1) = fu(i,j)
        fw(((i-1)*m+j)*2) = fv(i,j)
       end do
      end do
      
c			write (6,*) 'coeff, D1',t,D1max
			
      return
      end  
c--------------------------------------------------------
c     The subroutine FA (advection terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fa(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
		  write (6,*) 'warning, dummy function fa called!!'
			do i=1,neqn
			f(i)=0.0d0
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
      dimension y(npdes),f(npdes),frjac(npdes,npdes)
      common /params/   K0,Z0,ns,nssq,is_rec
			double precision K0,Z0
			logical is_frjac
			
			m=ns
			k=(ieqn+1)/2
			i=k/m+1
			j=mod(k,m)
			h=1.d0/m
      xx = (i-0.5d0)*h
      yy = (j-0.5d0)*h
      Z=1.d0
			qq=1.d0/6.d0
      if (abs(xx-0.5d0).le.qq.and.abs(yy-0.5d0).le.qq)  then
			Z = Z0	
c		  write (6,*) '!R',i,j,((i-1)*m+j)*2-1
			end if
      sigma = Z**3/y(2)**3
      f(1)= sigma*(y(2)**4 - y(1))
			f(2)=-f(1)
			if (is_frjac) then
			frjac(1,1)=-sigma
			frjac(2,1)=sigma
			frjac(1,2)=Z**3*(1.d0+y(1)*3.d0/y(2)**4)
			frjac(2,2)=-frjac(1,2)
			end if
			return
      end  
c--------------------------------------------------------
c     The subroutine FW (noise terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fw(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
		  write (6,*) 'warning, dummy function fw called!!'
		  do i=1,neqn
			f(i)=0.d0
			end do
		  return
      end  
c--------------------------------------------------------
c     The subroutine FD2 (non-symmetric diffusion)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fd2(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
		  write (6,*) 'warning, dummy function fd2 called !!'
			do i=1,neqn
			f(i)=0.0d0
			end do
			return
      end  