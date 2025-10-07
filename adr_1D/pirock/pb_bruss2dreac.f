c--------------------------------------------------------
c    2D Brusselator problem with stiff reaction
c--------------------------------------------------------
c    This driver shows how to use PIROCK. It solves a
c    system of ODEs resulting from the 2-dimensional space 
c    discretization of the Brusselator equations (u=u(x,y,t),v=v(x,y,t)):
c     
c    u_t=1.3+u^2*v-(1.+2.e7)*u+0.1*(u_{xx}+u_{yy})
c    v_t=2.e7*u-u^2*v+0.1*(v_{xx}+v_{yy})     for t>=0, 0<= x <= 1, 0<= y <= 1
c
c    with initial conditions 
c
c    u(x,y,0)=22*y*(1-y)^{3/2}  v(x,y,0)=27*x*(1-x)^{3/2}
c
c    and periodic boundary conditions
c
c    u(x+1,y,t)=u(x,y,t),  v(x,y+1,t)=v(x,y,t).
c
c    We discretize the space variables with
c    x_i=i/(N+1), y_i=i/(N+1) for i=0,1,...,N,
c    with N=200. We obtain a system of 80000
c    equations. 
c
c--------------------------------------------------------

      SUBROUTINE init(nsd,t,tend,y)
			implicit double precision (a-h,o-z)
      double precision  y(*)
			
c --- common parameters for the problem -----
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth

c ----- dimensions -----
			neqn=nsd*nsd*npdes
		  ns=nsd
      nssq=ns*ns
      nsnsm1=ns*(ns-1)
      alf=1.d-1
			amult=0.d0
			uxadv=-.5d-1  *amult
			uyadv=0.5d-1  *amult
			vxadv=.4d-1   *amult
			vyadv=.3d-1   *amult
      alph=alf
			brussa=1.3d0
			brussb=2.0d7
c			brussb=1.0d0

      write(6,*) 'Integration of the '
     &   ,'2-dim Brusselator problem, ns=',ns
c ----- initial and end point of integration -----
      t=0.0d0
      tend=2.d0
c ----- initial values -----
      ans=ns
      do j=1,ns
        yy=(j-1)/ans
        do i=1,ns
          y(((j-1)*ns+i)*2-1)=22.d0*yy*(1.d0-yy)**(1.5d0)
        end do
      end do
      do i=1,ns
        xx=(i-1)/ans
        do j=1,ns
           y(((j-1)*ns+i)*2)=27.d0*xx*(1.d0-xx)**(1.5d0)
        end do
      end do
			
      return
      end
			
      SUBROUTINE solout(neqn,t,tend,y,ytmp)
			implicit double precision (a-h,o-z)
      double precision  y(neqn),ytmp(neqn)
			
c --- common parameters for the problem -----
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth

c ----- file for solution -----
       open(8,file='sol.dat')
       rewind 8
		  write (8,*) t,((y(((j-1)*ns+i)*2-1),i=1,ns),j=1,ns),
     &     ((y(((j-1)*ns+i)*2),i=1,ns),j=1,ns)

			 write(6,*) 'Solution is tabulated in file sol.dat'
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
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
        rhodiff = 8.0d0*nssq*alf + 2.d0 
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
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
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
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
c ----- constants for inhomogenity -----
      ans=ns
      radsq=0.1d0**2
      if(x.ge.1.1d0)then
        bet=5.00d0				
      else
        bet=0.00d0
      end if
c ----- big loop -----
      do i=1,nssq
c ----- left neighbour -----
         if(mod(i,ns).eq.1)then
            uleft=y((i+ns-1)*2-1)
            vleft=y((i+ns-1)*2)
         else
            uleft=y((i-1)*2-1)
            vleft=y((i-1)*2)
         end if
c ----- right neighbour -----
         if(mod(i,ns).eq.0)then
            uright=y((i-ns+1)*2-1)
            vright=y((i-ns+1)*2)
         else
            uright=y((i+1)*2-1)
            vright=y((i+1)*2)
         end if
c ----- lower neighbour -----
         if(i.le.ns)then
            ulow=y((i+nsnsm1)*2-1)
            vlow=y((i+nsnsm1)*2)
         else
            ulow=y((i-ns)*2-1)
            vlow=y((i-ns)*2)
         end if
c ----- upper neighbour -----
         if(i.gt.nsnsm1)then
            uup=y((i-nsnsm1)*2-1)
            vup=y((i-nsnsm1)*2)
         else
            uup=y((i+ns)*2-1)
            vup=y((i+ns)*2)
         end if
c ----- the derivative -----
         uij=y(i*2-1)
         vij=y(i*2)
		     f(i*2-1)=alf*nssq*(uleft+uright+ulow+uup-4.d0*uij)
         f(i*2)=alf*nssq*(vleft+vright+vlow+vup-4.d0*vij)
      end do
      return
      end  
c--------------------------------------------------------
c     The subroutine FR (reaction terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fr(neqn,npdes,ieqn,x,y,f,frjac,is_frjac)
      implicit double precision (a-h,o-z)
      dimension y(npdes),f(npdes),frjac(npdes,npdes)
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
			logical is_frjac
			uij=y(1)
      vij=y(2)
      f(1)=brussa+uij*uij*vij-(brussb+1.d0)*uij 
      f(2)=brussb*uij - uij*uij*vij
			if (is_frjac) then
			frjac(1,1)=2.d0*uij*vij-(brussb+1.d0) 
			frjac(2,1)=brussb - 2.d0*uij*vij
			frjac(1,2)=uij*uij 
			frjac(2,2)=-uij*uij
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
		
		  do i=1,neqn
			f(i)=0.d0
			end do
		  return
      end  
c--------------------------------------------------------
c     The subroutine FA (advection terms)
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fa(neqn,x,y,f)
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/alf,ns,nssq,nsnsm1,nsm1sq,
     &    brussa,brussb,uxadv,vxadv,uyadv,vyadv,imeth
		  write (6,*) 'warning, dummy function fa called !!'
			do i=1,neqn
			f(i)=0.0d0
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