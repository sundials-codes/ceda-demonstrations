# Author: Sylvia Amihere @ UMBC

all: adr1D bruss2dadv bruss2dnoise bruss2dreac comb pides

adr1D: dr_adr_1D.f pb_adr_1D.f pirock.f decsol.f
	gfortran -std=legacy -o adr1D dr_adr_1D.f

bruss2dadv: dr_bruss2dadv.f pb_bruss2dadv.f pirock.f decsol.f
	gfortran -std=legacy -o bruss2dadv dr_bruss2dadv.f

bruss2dnoise: dr_bruss2dnoise.f pb_bruss2dnoise.f pirock.f decsol.f
	gfortran -std=legacy -o bruss2dnoise dr_bruss2dnoise.f

bruss2dreac: dr_bruss2dreac.f pb_bruss2dreac.f pirock.f decsol.f
	gfortran -std=legacy -o bruss2dreac dr_bruss2dreac.f

comb: dr_comb.f pb_comb.f pirock.f decsol.f
	gfortran -std=legacy -o comb dr_comb.f

pides: dr_pides.f pb_pides.f pirock.f decsol.f
	gfortran -std=legacy -o pides dr_pides.f

# # Clean
 EXECUTABLES = adr1D bruss2dadv bruss2dnoise bruss2dreac comb pides

.PHONY: clean
clean:
	rm -f *.o $(EXECUTABLES)
