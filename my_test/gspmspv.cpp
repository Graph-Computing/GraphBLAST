#define GRB_USE_SEQUENTIAL
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

int main( int argc, char** argv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool DEBUG = true;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols, 
        &nvals, 0, DEBUG);
  }

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  CHECK( a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL) );
  CHECK( a.nrows(&nrows) );
  CHECK( a.ncols(&ncols) );
  CHECK( a.nvals(&nvals) );
  if( DEBUG ) CHECK( a.print() );

  // Vector x
  graphblas::Vector<float> x(nrows);
  std::vector<graphblas::Index> x_ind = {1,   2,   3};
  std::vector<float>            x_val = {2.f, 2.f, 2.f};
  CHECK( x.build(&x_ind, &x_val, 3, GrB_NULL) );
  CHECK( x.size(&nrows) );
  if( DEBUG ) CHECK( x.print() );

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Vector mask
  graphblas::Vector<float> m(nrows);
  CHECK( m.fill(-1.f) );
  CHECK( m.setElement(0.f, 1) );
  CHECK( m.size(&nrows) );

  // Descriptor
  graphblas::Descriptor desc;
  CHECK( desc.set(graphblas::GrB_MASK, graphblas::GrB_SCMP) );
  CHECK( desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::vxm<float, float, float>(&y, &m, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);
  warmup.Stop();
 
  CpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  int NUM_ITER = 1;//0;
  for( int i=0; i<NUM_ITER; i++ )
  {
    graphblas::vxm<float, float, float>( &y, &m, GrB_NULL, 
        graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc );
  }
  //cudaProfilerStop();
  cpu_vxm.Stop();

  float flop = 0;
  if( DEBUG ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = cpu_vxm.ElapsedMillis();
  std::cout << "vxm, " << elapsed_vxm/NUM_ITER << "\n";

  if( DEBUG ) y.print();
  return 0;
}
