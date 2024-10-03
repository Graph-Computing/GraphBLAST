template <typename W, typename a, typename U, typename M,
          typename BinaryOpT, typename SemiringT>
Info mxv(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Matrix<a>* A,
         const Vector<U>* u,
         Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin mxv===\n";
    CHECK(u_t->print());
  }

  // Get storage:
  Storage u_vec_type;
  Storage A_mat_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(A->getStorage(&A_mat_type));

  // Transpose:
  Desc_value inp1_mode;
  CHECK(desc->get(GrB_INP1, &inp1_mode));
  if (inp1_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  LoadBalanceMode lb_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);
  if (desc->debug())
    std::cout << "Load balance mode: " << lb_mode << std::endl;

  // Conversions:
  SparseMatrixFormat A_format;
  bool A_symmetric;
  CHECK(A->getFormat(&A_format));
  CHECK(A->getSymmetry(&A_symmetric));

  Desc_value mxv_mode, tol;
  CHECK(desc->get(GrB_MXVMODE, &mxv_mode));
  CHECK(desc->get(GrB_TOL,     &tol));

  // Fallback for lacking CSC storage overrides any mxvmode selections
  if (!A_symmetric && A_format == GrB_SPARSE_MATRIX_CSRONLY) {
    if (u_vec_type == GrB_SPARSE)
      CHECK(u_t->sparse2dense(op.identity(), desc));
  } else if (mxv_mode == GrB_PUSHPULL) {
    CHECK(u_t->convert(op.identity(), desc->switchpoint(), desc));
  } else if (mxv_mode == GrB_PUSHONLY && u_vec_type == GrB_DENSE) {
    CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (mxv_mode == GrB_PULLONLY && u_vec_type == GrB_SPARSE) {
    CHECK(u_t->sparse2dense(op.identity(), desc));
  }

  // Check if vector type was changed due to conversion!
  CHECK(u->getStorage(&u_vec_type));

  // 3 cases:
  // 1) SpMSpV: SpMat x SpVec (preferred to 3)
  // 2) SpMV:   SpMat x DeVec
  // 3) SpMV:   SpMat x DeVec (fallback if CSC representation not available)
  // 4) GeMV:   DeMat x DeVec
  if (A_mat_type == GrB_SPARSE && u_vec_type == GrB_SPARSE) {
    // simple 和 twc 两种负载均衡算法尚未实现
    if (lb_mode == GrB_LOAD_BALANCE_SIMPLE ||
        lb_mode == GrB_LOAD_BALANCE_TWC) {
      CHECK(w->setStorage(GrB_DENSE));
      // 1a) Simple SpMSpV no load-balancing codepath
      if (lb_mode == GrB_LOAD_BALANCE_SIMPLE) {
        std::cout << "Simple SPMSPV not implemented yet!\n";
        return GrB_NOT_IMPLEMENTED;
        // CHECK( spmspvSimple(&w->dense_, mask, accum, op, &A->sparse_,
        //     &u->sparse_, desc) );
      // 1b) Thread-warp-block (single kernel) codepath
      } else if (lb_mode == GrB_LOAD_BALANCE_TWC) {
        std::cout << "Error: B40C load-balance algorithm not implemented yet!\n";
        return GrB_NOT_IMPLEMENTED;
      }
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
      CHECK(w->dense2sparse(op.identity(), desc));
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
    // 1c) Merge-path (two-phase decomposition) codepath
    // merge-path 负载均衡算法
    } else if (lb_mode == GrB_LOAD_BALANCE_MERGE) {
      CHECK(w->setStorage(GrB_SPARSE));
      CHECK(spmspvMerge(&w->sparse_, mask, accum, op, &A->sparse_,
          &u->sparse_, desc));
    } else {
      std::cout << "Error: Invalid load-balance algorithm!\n";
    }
    desc->lastmxv_ = GrB_PUSHONLY;
  } else {
    CHECK(w->sparse2dense(op.identity(), desc));
    if (A_mat_type == GrB_SPARSE) {
      CHECK(spmv(&w->dense_, mask, accum, op, &A->sparse_,
          &u->dense_, desc));
    } else {
      CHECK(gemv(&w->dense_, mask, accum, op, &A->dense_,
          &u->dense_, desc));
    }
    desc->lastmxv_ = GrB_PULLONLY;
  }

  if (desc->debug()) {
    std::cout << "===End mxv===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}