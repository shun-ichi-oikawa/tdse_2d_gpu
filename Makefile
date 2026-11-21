all: sch sch_fp sch_ufm

GENCODE = \
-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 \
-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 \
-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 \
-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86

NVCCFLAG = $(GENCODE) -lcudart -lcublas -lpthread -lgfortran

DEPS = sch.cu EXPECT.cu constant.cu func.cu potential.cu FDM.cu write.cu PHI_SOR.cu DEHINT.o Makefile \
       timer.c restart.c

sch: $(DEPS)
	nvcc -O3 -Xcompiler -Wall -o sch sch.cu DEHINT.o $(NVCCFLAG)

sch_fp: $(DEPS)
	nvcc -O3 -Xcompiler -Wall -o sch_fp sch.cu DEHINT.o $(NVCCFLAG) -DFieldParticle

sch_ufm: $(DEPS)
	nvcc -O3 -Xcompiler -Wall -o sch_ufm sch.cu DEHINT.o $(NVCCFLAG) -D_UFM_

DEHINT.o: DEHINT.for
	gfortran -O3 -c DEHINT.for

clean:
	rm -f sch sch_fp sch_ufm DEHINT.o
