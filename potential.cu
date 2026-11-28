#pragma once  // include guard for header file

__device__ __host__
static inline REAL X ( REAL x ) { return 0; }

__device__ __host__
static inline REAL Y ( REAL y ) {
	//const REAL LB_INV = 1e-6;	// inverse gradient scale length of magnetic field; 0 for uniform B
	//const REAL LB_INV = 1e-5;	// inverse gradient scale length of magnetic field; 0 for uniform B
	const REAL LB_INV = 0;	// inverse gradient scale length of magnetic field; 0 for uniform B
	return y * ( 1 - LB_INV / 2 * y);
}

__device__ __host__ void qA ( REAL x, REAL y, REAL Bz, REAL q, REAL* qAx, REAL* qAy )
{	*qAx = - q * Bz * Y(y);
	*qAy =   q * Bz * X(x);
}


#ifdef FieldParticle
extern "C"
{	void  hiab_();
	void  dehint_(REAL *, REAL *, REAL *, REAL *);
}

void PotEne
	(	REAL *u, const REAL *x, const REAL *y,
		const REAL q, const REAL sgm, const REAL Eps0, const int Nx2, const int Ny2
	)
{	const int  nf = 1;	// number of field particles
	REAL xf[nf], yf[nf], qf[nf];
	REAL zero = 0;		// lower bound of integration (integration interval is [a,b]=[0, infinity])
	REAL EPS  = 1e-12;	// absolute error tolerance
	REAL V;

	xf[0] = 0; yf[0] = 0; qf[0] = 3e-5;
	hiab_();
	for(		int iy = 0; iy < Ny2; iy++) { REAL y_ij = y[iy];
		for(	int ix = 0; ix < Nx2; ix++) { REAL x_ij = x[ix];
			int II = ix + Nx2 * iy;
			REAL phi_ij = 0;
			for( int f = 0; f < nf; f++)
			{	REAL R = sqrt( Square( x_ij - xf[f] ) + Square( y_ij - yf[f] ) );
				REAL eta = R / sgm;
				dehint_ ( &eta, &zero, &EPS, &V );
				phi_ij += qf[f] * V;
			}
			phi_ij =     phi_ij / ( Square( pi ) * Eps0 * sgm );
			u[II]  = q * phi_ij;
		}
	}
}

#elif defined(StaticDebye)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

void PotEne
	(	REAL *u, const REAL *x, const REAL *y,
		const REAL q, const REAL sgm, const REAL Eps0, const int Nx2, const int Ny2
	)
{
	// read 1332-body binary file (63944 bytes, ordered as x,y,z,u,v,w)
	const char* fname = "0001332_10^0.bbb"; // <- replace with actual binary filename
	FILE* fp = std::fopen(fname, "rb");
	if (!fp) {
		std::fprintf(stderr, "Error: Cannot open %s\n", fname);
		std::exit(1);
	}

	int32_t header = 0, footer = 0;
	const int N_TOTAL = 1332;
	std::vector<double> buf(N_TOTAL * 6);

	std::fread(&header, sizeof(int32_t), 1, fp);
	std::fread(buf.data(), sizeof(double), N_TOTAL * 6, fp);
	std::fread(&footer, sizeof(int32_t), 1, fp);
	std::fclose(fp);

	const REAL L_SCALE = 20.6; // mean inter-proton distance d_tilde
	const REAL qf_each = 3e-5; // test with perturbation scale 3e-5 matching JMP (modify if needed)
	const REAL sgm2    = Square(sgm);
	const REAL coeff   = q * qf_each / (4.0 * pi * Eps0);

	// skip the first particle (i=0, stationary test particle at origin) and extract i=1..1331
	const int nf = N_TOTAL - 1;
	std::vector<REAL> xf(nf), yf(nf), zf(nf);
	for (int f = 0; f < nf; f++) {
		int idx = f + 1;
		xf[f] = static_cast<REAL>(buf[6 * idx + 0]) * L_SCALE;
		yf[f] = static_cast<REAL>(buf[6 * idx + 1]) * L_SCALE;
		zf[f] = static_cast<REAL>(buf[6 * idx + 2]) * L_SCALE;
	}

	// compute static potential array u[II] at all grid points (x[ix], y[iy], 0)
	#pragma omp parallel for collapse(2)
	for (int iy = 0; iy < Ny2; iy++) {
		for (int ix = 0; ix < Nx2; ix++) {
			REAL x_ij = x[ix];
			REAL y_ij = y[iy];
			int II = ix + Nx2 * iy;
			REAL v_sum = 0;
			for (int f = 0; f < nf; f++) {
				REAL r_soft = sqrt(Square(x_ij - xf[f]) + Square(y_ij - yf[f]) + Square(zf[f]) + sgm2);
				v_sum += 1.0 / r_soft;
			}
			u[II] = coeff * v_sum;
		}
	}
	std::printf("PotEne (StaticDebye): computed %d particles onto %dx%d grid.\n", nf, Nx2, Ny2);
}
#elif defined(DynamicDebye)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

constexpr int N_BG_DYN = 1331; // full NL=5 domain (1332 - 1)
static REAL *d_bg_x = nullptr, *d_bg_y = nullptr, *d_bg_z0 = nullptr, *d_bg_vz = nullptr;
static REAL *d_grid_x = nullptr, *d_grid_y = nullptr;

// transfer initial positions and velocities to GPU memory (called once at start of main)
void PotEne
	(	REAL *u, const REAL *x, const REAL *y,
		const REAL q, const REAL sgm, const REAL Eps0, const int Nx2, const int Ny2
	)
{
	const char* fname = "0001332_10^0.bbb";
	FILE* fp = std::fopen(fname, "rb");
	if (!fp) {
		std::fprintf(stderr, "Error: Cannot open %s\n", fname);
		std::exit(1);
	}

	int32_t header = 0, footer = 0;
	const int N_TOTAL = 1332;
	std::vector<double> buf(N_TOTAL * 6);
	std::fread(&header, sizeof(int32_t), 1, fp);
	std::fread(buf.data(), sizeof(double), N_TOTAL * 6, fp);
	std::fread(&footer, sizeof(int32_t), 1, fp);
	std::fclose(fp);

	const REAL L_SCALE = 20.6;
	const REAL V_SCALE = 1.38e5;

	std::vector<REAL> h_x(N_BG_DYN), h_y(N_BG_DYN), h_z0(N_BG_DYN), h_vz(N_BG_DYN);
	for (int f = 0; f < N_BG_DYN; f++) {
		int idx = f + 1;
		h_x[f]  = static_cast<REAL>(buf[6 * idx + 0]) * L_SCALE;
		h_y[f]  = static_cast<REAL>(buf[6 * idx + 1]) * L_SCALE;
		h_z0[f] = static_cast<REAL>(buf[6 * idx + 2]) * L_SCALE;
		h_vz[f] = static_cast<REAL>(buf[6 * idx + 5]) * V_SCALE;
	}

	cudaMalloc(&d_bg_x,  sizeof(REAL) * N_BG_DYN);
	cudaMalloc(&d_bg_y,  sizeof(REAL) * N_BG_DYN);
	cudaMalloc(&d_bg_z0, sizeof(REAL) * N_BG_DYN);
	cudaMalloc(&d_bg_vz, sizeof(REAL) * N_BG_DYN);
	cudaMalloc(&d_grid_x, sizeof(REAL) * Nx2);
	cudaMalloc(&d_grid_y, sizeof(REAL) * Ny2);

	cudaMemcpy(d_bg_x,  h_x.data(),  sizeof(REAL) * N_BG_DYN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bg_y,  h_y.data(),  sizeof(REAL) * N_BG_DYN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bg_z0, h_z0.data(), sizeof(REAL) * N_BG_DYN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bg_vz, h_vz.data(), sizeof(REAL) * N_BG_DYN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_x, x, sizeof(REAL) * Nx2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_y, y, sizeof(REAL) * Ny2, cudaMemcpyHostToDevice);

	std::printf("PotEne (DynamicDebye): initialized %d moving particles on GPU.\n", N_BG_DYN);
// t = 0 initial potential
	const REAL coeff = q * 1.0 / (4.0 * pi * Eps0);
	const REAL sgm2  = Square(sgm);
	for (int iy = 0; iy < Ny2; iy++) {
		for (int ix = 0; ix < Nx2; ix++) {
			REAL v_sum = 0.0;
			for (int f = 0; f < N_BG_DYN; f++) {
				REAL r_soft = sqrt(Square(x[ix] - h_x[f]) + Square(y[iy] - h_y[f]) + Square(h_z0[f]) + sgm2);
				v_sum += 1.0 / r_soft;
			}
			u[ix + Nx2 * iy] = coeff * v_sum;
		}
	}
}

// GPU kernel to update u_device across all grid points at midpoint time t_mid
__global__ void CalcDynamicPotKernel
	(	REAL *u_dev, const REAL *gx, const REAL *gy,
		const REAL *bx, const REAL *by, const REAL *bz0, const REAL *bvz,
		const REAL coeff, const REAL sgm2, const REAL t_mid, const REAL L_box,
		const int Nx2, const int Ny2
	)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix >= Nx2 || iy >= Ny2) return;

	REAL x_ij = gx[ix];
	REAL y_ij = gy[iy];
	REAL v_sum = 0.0;
	const REAL V_ELE_RATIO = 42.85; // sqrt(mp / me) ~= 42.85 (thermal velocity ratio of background electrons)

	for (int f = 0; f < N_BG_DYN; f++) {
		REAL dx = x_ij - bx[f];
		REAL dy = y_ij - by[f];

		// 1. contribution from background protons (+qf_each)
		REAL dz_p = remainder(bz0[f] + bvz[f] * t_mid, L_box);
		REAL r_p  = sqrt(dx * dx + dy * dy + dz_p * dz_p + sgm2);
		v_sum += 1.0 / r_p;

		// 2. contribution from background electrons (-qf_each) (counter-streaming at 42.85x thermal speed, quasi-neutrality)
		REAL dz_e = remainder(bz0[f] - (bvz[f] * V_ELE_RATIO) * t_mid, L_box);
		REAL r_e  = sqrt(dx * dx + dy * dy + dz_e * dz_e + sgm2);
		v_sum -= 1.0 / r_e;
	}
	u_dev[ix + Nx2 * iy] = coeff * v_sum;
}

inline void UpdateDynamicPot(REAL *u_dev, REAL q, REAL sgm, REAL Eps0, int Nx2, int Ny2, REAL t_mid)
{
	const REAL qf_each = 3e-5; // true elementary proton charge q_f = 1.0
	const REAL coeff   = q * qf_each / (4.0 * pi * Eps0);
	const REAL sgm2    = Square(sgm);
	const REAL L_box   = 11.0 * 20.6; // periodic box length for NL=5 (11 cells)

	dim3 blk(16, 16);
	dim3 grd((Nx2 + 15) / 16, (Ny2 + 15) / 16);
	CalcDynamicPotKernel<<<grd, blk>>>(
		u_dev, d_grid_x, d_grid_y,
		d_bg_x, d_bg_y, d_bg_z0, d_bg_vz,
		coeff, sgm2, t_mid, L_box, Nx2, Ny2
	);
}

#else
	__device__ __host__ REAL potential ( REAL x, REAL y, REAL q )
	{
	//	return 0;
		#ifndef _UFM_
		const REAL LE4_INV = 1e-6;
		#else
		const REAL LE4_INV = 0;
		#endif
		const REAL Ey = 1;
	//	REAL y2 = y * y;
		return - q * Ey * y * LE4_INV;
	//	REAL Ey = 0;
	//	REAL Ey = 1;
	//	REAL Ey = 1e-4;
	//	REAL Ey = 1e-5;
	//	REAL Ey = 1e-6;
		//REAL LE_INV = 0;

	//	REAL k_E = 0.1;
	//	REAL k_E = 0.5;
	//	REAL k_E = 1;
	//	REAL k_E = 2;
	//	REAL k_E   = 1e-4;
	//	REAL lmd_E = 1e+4; // lmd_E = 1 / k_E

	/* 1. linear type */
		//return - q * Ey * y *(1 - LE_INV / 2 * y);
	/* 3. sinusoidal type E ~ + cos */
		//return - q * Ey / k_E * sin(k_E * y);
		/* 3+. rewrite 1/k_E -> lmd_E to avoid division by a small number */
		//return - q * Ey * lmd_E * sin(y / lmd_E);
	/* 3+.sinusoidal type E ~ + sin */
	//	return + q * Ey / k_E * cos(k_E * y);
	}
#endif

