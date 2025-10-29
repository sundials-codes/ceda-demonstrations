
c    Programmer(s): Sylvia Amihere @ UMBC
c    Based on the Based on the SUNDIALS example ark_advection_diffusion_reaction.cpp by
c    David J. Gardner @ LLNL and Daniel Reynolds @ UMBC
c-------------------------------------------------------------------------------------------
c    1D Brusselator problem with stiff reaction and advection
c----------------------------------------------------------------------------------------------
c    This driver shows how to use PIROCK. It solves a
c    system of ODEs resulting from the 1-dimensional space
c    discretization of the Brusselator equations (u=u(x,t),v=v(x,t), w=w(x,t)):
c
c    This example simulates the 1D advection-diffusion-reaction equation,
c
c           u_t = -c u_x + d u_xx + A - (w + 1) * u + v * u^2
c           v_t = -c v_x + d v_xx + w * u - v * u^2
c           w_t = -c w_x + d w_xx + (B - w) / eps - w * u
c
c    where u, v, and w represent the concentrations of chemical species, c = 0.01
c    is the advection speed, d = 0.1 is the diffusion rate, and the species with
c    constant concentration over time are A = 0.6 and B = 2.0.
c
c    The problem is evolved for t in [0, 3] and x in [0, 1], with initial conditions given by
c
c            u(0,x) =  A  + 0.1 * sin(pi * x)
c            v(0,x) = B/A + 0.1 * sin(pi * x)
c            w(0,x) =  B  + 0.1 * sin(pi * x)
c
c    and stationary boundary conditions i.e.,
c
c           u_t(t,0) = u_t(t,1) = 0,
c           v_t(t,0) = v_t(t,1) = 0,
c           w_t(t,0) = w_t(t,1) = 0.
c
c    We discretize the space variables with x_i=(i-1)/(N-1), for i=1,...,N, with N=512.
c    We obtain a system of 3N equations.
c--------------------------------------------------------------------------------------------

      SUBROUTINE init(nsd,t,tend,y)
	implicit double precision (a-h,o-z)
      double precision  y(*)
      double precision  pi, xx
      parameter(pi = 3.141592653589793d0)


c --- common parameters for the problem -----
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21

c ----- dimensions -----
        neqn   = nsd*npdes
	  ns     = nsd
        nssq   = (ns-1)*(ns-1)
        nsnsm1 = ns*(ns-1)

c c --- read input from namelist file (if it exists) ---
c         open(10, file='namelist_read.txt', status='old', err=100)
c         read(10, nml=list1)
c         close(10)
c         goto 110

c   100   continue
c         write(6,*) 'Could not open namelist file'

        write(6,*) 'Integration of the '
     &   ,'1-dim Brusselator problem, ns=',ns
        write(6,*) 'advection pb:', uxadv,vxadv,wxadv
        write(6,*) 'diffusion pb:', alf
        write(6,*) 'brusselator A:', brussa
        write(6,*) 'brusselator B:', brussb
        write(6,*) 'brusselator eps:', eps
c --------------- multiplying by 1.d0 because of tests that are run from python script
c --------------- because Python can't take in values with '.d'
        alf=alf*1.d0
        amult=amult*1.d0
        uxadv=uxadv*1.d0
        uyadv=uyadv*1.d0
        vxadv=vxadv*1.d0
        vyadv=vyadv*1.d0
        wxadv=wxadv*1.d0
        wyadv=wyadv*1.d0
        brussa=brussa*1.d0
        brussb=brussb*1.d0
        eps=eps*1.d0
c        stop

c ----- initial and end point of integration -----
        t = 0.0d0
        tend = 3.d0

c ----- initial values -----
        do i=1,ns
            xx       = ((i-1.d0)/(ns-1.d0))
            y(i*3-2) = brussa          + (1.d-1)*SIN(pi*xx)
            y(i*3-1) = (brussb/brussa) + (1.d-1)*SIN(pi*xx)
            y(i*3)   = brussb          + (1.d-1)*SIN(pi*xx)
        end do

	  radadv=rhoadv(neqn,t,y)
	  write (6,*) 'amult',amult,'adv spectral radius',radadv
        return
      end

c--------------------------------------------------------
c     Solution is saved in a .dat file
c--------------------------------------------------------
      SUBROUTINE solout(neqn,t,tend,y,ytmp)
	implicit double precision (a-h,o-z)
      double precision  y(neqn),ytmp(neqn)

c --- common parameters for the problem -----
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21

c ----- file for solution -----
      open(8,file='sol.dat')
      rewind 8
	write (8,*) t,(y(i*3-2),i=1,ns),
     &      (y(i*3-1),i=1,ns), (y(i*3),i=1,ns)

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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
      rhodiff = 4.0d0*(nssq)*alf
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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
      rhoadv = (abs(uxadv)+abs(vxadv)+abs(wxadv))*(ns-1)
      return
      end
c--------------------------------------------------------
c     The subroutine FBRUS compute the value of f(x,y) and
c     has to be declared as external.
c--------------------------------------------------------
      subroutine fd(neqn,x,y,f)
c ----- brusselator with diffusion in 1 dim. space -----
      implicit double precision (a-h,o-z)
      dimension y(neqn),f(neqn)
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
c ----- zero boundary conditions -----
      f(1)      = 0.d0
      f(2)      = 0.d0
      f(3)      = 0.d0
      f(3*ns-2) = 0.d0
      f(3*ns-1) = 0.d0
      f(3*ns)   = 0.d0
c ----- big loop -----
      do i=2,ns-1
c ----- left neighbour -----
          uleft=y((i-1)*3-2)
          vleft=y((i-1)*3-1)
          wleft=y((i-1)*3)
c ----- right neighbour -----
          uright=y((i+1)*3-2)
          vright=y((i+1)*3-1)
          wright=y((i+1)*3)
c ----- the derivative -----
          uij=y(i*3-2)
          vij=y(i*3-1)
          wij=y(i*3)
          f(i*3-2) = alf * nssq * (uleft + uright - 2.d0*uij)
          f(i*3-1) = alf * nssq * (vleft + vright - 2.d0*vij)
          f(i*3)   = alf * nssq * (wleft + wright - 2.d0*wij)
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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21

c ----- zero boundary conditions -----
      f(1)      = 0.d0
      f(2)      = 0.d0
      f(3)      = 0.d0
      f(3*ns-2) = 0.d0
      f(3*ns-1) = 0.d0
      f(3*ns)   = 0.d0
c ----- big loop -----
      do i=2,ns-1
c ----- left neighbour -----
          uleft=y((i-1)*3-2)
          vleft=y((i-1)*3-1)
          wleft=y((i-1)*3)
c ----- right neighbour -----
          uright=y((i+1)*3-2)
          vright=y((i+1)*3-1)
          wright=y((i+1)*3)
c ----- the derivative -----
          uij=y(i*3-2)
          vij=y(i*3-1)
          wij=y(i*3)
	    f(i*3-2)=0.5d0*(ns-1)*(-uxadv*(uright-uleft))
          f(i*3-1)=0.5d0*(ns-1)*(-vxadv*(vright-vleft))
          f(i*3)  =0.5d0*(ns-1)*(-wxadv*(wright-wleft))
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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
      write (6,*) 'warning, dummy function fd2 called !!'
	do i=1,neqn
	    f(i)=0.0d0
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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
	logical is_frjac
      uij = y(1)
      vij = y(2)
      wij = y(3)

      if ((ieqn .eq. 1) .or. (ieqn .eq. neqn-2)) then
          f(1) = 0.d0
          f(2) = 0.d0
          f(3) = 0.d0
      else
          f(1) = brussa - (wij+1.d0)*uij + vij*uij*uij
          f(2) = wij*uij - vij*uij*uij
          f(3) = (brussb - wij)/eps - wij*uij
      end if

      if (is_frjac) then
          if ((ieqn .eq. 1) .or. (ieqn .eq. neqn-2)) then
              frjac(1,1) = 0.d0
              frjac(2,1) = 0.d0
              frjac(3,1) = 0.d0

              frjac(1,2) = 0.d0
              frjac(2,2) = 0.d0
              frjac(3,2) = 0.d0

              frjac(1,3) = 0.d0
              frjac(2,3) = 0.d0
              frjac(3,3) = 0.d0
          else
              frjac(1,1) = -(wij+1.d0) + 2.d0*uij*vij
              frjac(1,2) = uij*uij
              frjac(1,3) = -uij

              frjac(2,1) = wij - 2.d0*uij*vij
              frjac(2,2) = -uij*uij
              frjac(2,3) = uij

              frjac(3,1) = -wij
              frjac(3,2) = 0.d0
              frjac(3,3) = -(1.d0/eps) - uij
          end if
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
      common/trans/alf,amult,ns,nssq,nsnsm1,nsm1sq,eps,atol,rtol,
     &    brussa,brussb,uxadv,vxadv,wxadv,uyadv,vyadv,wyadv,imeth,iwork20,iwork21
	write (6,*) 'WARNING DUMMY FUNCTION FW CALLED'
	do i=1,neqn
	    f(i)=0.d0
	end do
	return
      end