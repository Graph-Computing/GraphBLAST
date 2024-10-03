template <typename W, typename U, typename a, typename M,
          typename BinaryOpT, typename SemiringT>
Info vxm(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Vector<U>* u,
         const Matrix<a>* A,
         Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin vxm===\n";
    CHECK(u_t->print());
  }

  // Get storage
  Storage u_vec_type;
  Storage A_mat_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(A->getStorage(&A_mat_type));

  // Transpose
  Desc_value inp0_mode;
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  if (inp0_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  // Treat vxm as an mxv with transposed matrix
  CHECK(desc->toggle(GrB_INP1));

  LoadBalanceMode lb_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);

  // Conversions
  // TODO(@ctcyang): add tol
  SparseMatrixFormat A_format;
  bool A_symmetric;
  CHECK(A->getFormat(&A_format));
  CHECK(A->getSymmetry(&A_symmetric));

  Desc_value vxm_mode, tol;
  CHECK(desc->get(GrB_MXVMODE, &vxm_mode));
  CHECK(desc->get(GrB_TOL,     &tol));
  if (desc->debug()) {
    std::cout << "Load balance mode: " << lb_mode << std::endl;
    std::cout << "Identity: " << op.identity() << std::endl;
    std::cout << "Sparse format: " << A_format << std::endl;
    std::cout << "Symmetric: " << A_symmetric << std::endl;
  }

  // Fallback for lacking CSC storage overrides any mxvmode selections
  if (!A_symmetric && A_format == GrB_SPARSE_MATRIX_CSRONLY) {
    if (u_vec_type == GrB_DENSE)
      CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PUSHPULL) {
    CHECK(u_t->convert(op.identity(), desc->switchpoint(), desc));
  } else if (vxm_mode == GrB_PUSHONLY && u_vec_type == GrB_DENSE) {
    CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PULLONLY && u_vec_type == GrB_SPARSE) {
    CHECK(u_t->sparse2dense(op.identity(), desc));
  }

  // Check if vector type was changed due to conversion!
  CHECK(u->getStorage(&u_vec_type));

  if (desc->debug())
    std::cout << "u_vec_type: " << u_vec_type << std::endl;

  // Breakdown into 4 cases:
  // 1) SpMSpV: SpMat x SpVec
  // 2) SpMV:   SpMat x DeVec (preferred to 3)
  // 3) SpMSpV: SpMat x SpVec (fallback if CSC representation not available)
  // 4) GeMV:   DeMat x DeVec
  //
  // Note: differs from mxv, because mxv would say instead:
  // 3) "... if CSC representation not available ..."
  if (A_mat_type == GrB_SPARSE && u_vec_type == GrB_SPARSE) {
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
      } else if (lb_mode == GrB_LOAD_BALANCE_TWC)
        std::cout << "Error: B40C load-balance algorithm not implemented yet!\n";
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
    } else if (lb_mode == GrB_LOAD_BALANCE_MERGE) {
      CHECK(w->setStorage(GrB_SPARSE));
      CHECK(spmspvMerge(&w->sparse_, mask, accum, op, &A->sparse_,
          &u->sparse_, desc));
    } else {
      std::cout << "Error: Invalid load-balance algorithm!\n";
    }
    desc->lastmxv_ = GrB_PUSHONLY;
  } else {
    // TODO(@ctcyang): Some performance left on table, sparse2dense should
    // only convert rather than setStorage if accum is being used
    CHECK(w->setStorage(GrB_DENSE));
    // CHECK(w->sparse2dense(op.identity(), desc));
    if (A_mat_type == GrB_SPARSE)
      CHECK(spmv(&w->dense_, mask, accum, op, &A->sparse_, &u->dense_,
          desc));
    else
      CHECK(gemv(&w->dense_, mask, accum, op, &A->dense_, &u->dense_,
          desc));
    desc->lastmxv_ = GrB_PULLONLY;
  }

  // Undo change to desc by toggling again
  CHECK(desc->toggle(GrB_INP1));

  if (desc->debug()) {
    std::cout << "===End vxm===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}