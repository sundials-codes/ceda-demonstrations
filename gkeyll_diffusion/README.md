Gkeyll diffusion example
------------------------

The program 'gk_diffusion_1x1v_p1.c' here is a minimum working example of how we would solve the equation
```math
\frac{\partial f(x,v_\parallel)}{\partial t} = \frac{\partial}{\partial x} D(x)\frac{\partial f(x,v_\parallel)}{\partial x}
```
in gkylzero with an infrastructure that mimics the gyrokinetic app.

To compile and run this program:
1. Install [gkylzero](https://github.com/ammarhakim/gkylzero) in the **gk-g0-app** branch.
2. Copy the Makefile in <install_dir>/gkylsoft/gkylzero/share into this folder.
3. Rename all the instances of `rt_vlasov_twostream_p2` in Makefile to `gk_diffusion_1x1v_p1`.
4. Run `make`.
5. Run the executable `./gk_diffusion_1x1v_p1`.

To plot the result of this simulation one can use [postgkyl](https://gkeyll.readthedocs.io/en/latest/postgkyl/main.html) (see sections on installing postgkyl and the postgkyl reference) in command line or in script mode. An example of a Python script can be found in this directory (`post_gk_diffusion_1x1v_p1.py`). Some examples of command line usage include: 

- Plot solution at vpar=0, overlaying each slice in time.
```
pgkyl "gk_diffusion_1x1v_p1-f_[0-9]*.gkyl" interp sel --z1 0. collect pl --lineouts 1 -x 'x' -y '$f(v_\parallel=0)$' --clabel 'Time'
````

- Movie of the solution at vpar=0 in time:
```
pgkyl "gk_diffusion_1x1v_p1-f_[0-9]*.gkyl" interp sel --z1 0. anim -x 'x' -y '$f(v_\parallel=0)$'
```

- Plot the L2 norm of the solution in time.
```
pgkyl gk_diffusion_1x1v_p1-f_L2.gkyl pl -x 'Time' -y '$\int dx\,|f|^2$'
```
