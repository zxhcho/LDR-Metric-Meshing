#pragma once
#include "mfem.hpp"

using namespace std;
using namespace mfem;

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

class InnerIterativeSolver : public mfem::Solver {
 private:
  CGSolver cg;
  GSSmoother prec;

 public:
  InnerIterativeSolver(const mfem::SparseMatrix& A) : mfem::Solver(A.Height()) {
    // 1. Setup Inner Preconditioner
    prec.SetOperator(A);

    // 2. Setup Inner Solver
    cg.SetOperator(A);
    cg.SetPreconditioner(prec);

    // --- OPTIMIZATION FOR ANISOTROPIC MESHES ---

    // Disable "smart" stopping criteria.
    // We force it to run exactly 'MaxIter' steps every time.
    // This ensures the operator is Linear (required for Outer GMRES)
    // and predictable in runtime.
    cg.SetRelTol(0.0);
    cg.SetAbsTol(0.0);

    // Budget: 20-30 iterations is usually the "sweet spot" for 2D.
    // For 3D or extremely stretched meshes, you might need 50.
    cg.SetMaxIter(30);

    // Silence the "No convergence!" warning.
    // We EXPECT it not to converge fully; that's fine for a preconditioner.
    cg.SetPrintLevel(-1);
  }

  void Mult(const mfem::Vector& r, mfem::Vector& z) const override {
    z = 0.0;
    // This will run exactly 30 iterations and return the best approximation
    // found.
    cg.Mult(r, z);
  }

  void SetOperator(const mfem::Operator& op) override {}
};

class FixedEigenSolver : public mfem::Solver
{
private:
  // RowMajor is efficient for converting from MFEM's CSR
  using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  Eigen::SimplicialLLT<SpMat> solver;
  int height;

public:
  FixedEigenSolver(const mfem::SparseMatrix &A) : mfem::Solver(A.Height())
  {
    height = A.Height();

    // --- ROBUST FIX: Use Triplets to Enforce Sorting ---
    // MFEM's 'EliminateEssentialBC' often leaves column indices unsorted.
    // Eigen's Map/Copy crashes on unsorted inputs.
    // Converting via Triplets forces a clean sort and rebuild.

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.NumNonZeroElems());

    const int *I = A.GetI();
    const int *J = A.GetJ();
    const double *V = A.GetData();

    for (int r = 0; r < height; ++r)
    {
      for (int k = I[r]; k < I[r + 1]; ++k)
      {
        triplets.emplace_back(r, J[k], V[k]);
      }
    }

    // Construct valid sorted matrix from triplets
    SpMat A_safe(height, A.Width());
    A_safe.setFromTriplets(triplets.begin(), triplets.end());

    // Compute Cholesky Factorization
    solver.compute(A_safe);

    if (solver.info() != Eigen::Success)
    {
      mfem::err << "Eigen Cholesky decomposition failed!" << std::endl;
      // Fallback: If numerical issue, try adding small diagonal shift?
      // For now just warn.
    }
  }

  void Mult(const mfem::Vector &b, mfem::Vector &x) const override
  {
    // Map MFEM vectors to Eigen (Zero Copy)
    Eigen::Map<const Eigen::VectorXd> b_eig(b.GetData(), height);
    Eigen::Map<Eigen::VectorXd> x_eig(x.GetData(), height);

    x_eig = solver.solve(b_eig);
  }

  void SetOperator(const mfem::Operator &op) override {}
};

// Custom Block Triangular Preconditioner
// Solves Pz = r, where P is Lower Triangular
class BlockTriangularPrec : public mfem::Solver {
 private:
  FixedEigenSolver& A00_inv;  // The Cholesky Solver (Exact Inverse of A00)
  SparseMatrix& A10;          // The Coupling Matrix (B)
  SparseMatrix& A11;          // The Diagonal Matrix (C)
  Array<int>& offsets;
  mutable Vector temp_r;  // Scratch memory

 public:
  BlockTriangularPrec(FixedEigenSolver& s00, SparseMatrix& s10,
                      SparseMatrix& s11, Array<int>& off)
      : mfem::Solver(off.Last()),
        A00_inv(s00),
        A10(s10),
        A11(s11),
        offsets(off) {
    temp_r.SetSize(offsets[2] - offsets[1]);
  }

  void SetOperator(const mfem::Operator& op) override {}

  // The "Magic" Step: Forward Substitution
  // 1. Solve u block
  // 2. Update residual for psi block using u
  // 3. Solve psi block
  void Mult(const mfem::Vector& r, mfem::Vector& z) const override {
    // Map MFEM sub-vectors (Zero Copy)
    Vector r_u(r.GetData() + offsets[0], offsets[1] - offsets[0]);
    Vector r_p(r.GetData() + offsets[1], offsets[2] - offsets[1]);

    Vector z_u(z.GetData() + offsets[0], offsets[1] - offsets[0]);
    Vector z_p(z.GetData() + offsets[1], offsets[2] - offsets[1]);

    // STEP 1: Solve Displacement EXACTLY
    // z_u = A00^-1 * r_u
    A00_inv.Mult(r_u, z_u);

    // STEP 2: Compute Coupling Update
    // The lower block equation is: A10 * z_u + A11 * z_p = r_p
    // So: A11 * z_p = r_p - A10 * z_u

    temp_r = r_p;
    A10.AddMult(z_u, temp_r, -1.0);  // temp = r_p - A10 * z_u

    // STEP 3: Solve Constraint EXACTLY
    // z_p = A11^-1 * temp
    // A11 is diagonal, so we divide by entries.
    double* d_A11 = A11.GetData();
    double* d_z = z_p.GetData();
    double* d_temp = temp_r.GetData();
    int size = z_p.Size();

    for (int i = 0; i < size; i++) {
      // Prevent division by zero (though A11 should be strictly negative here)
      double val = d_A11[i];
      if (std::abs(val) < 1e-16) val = -1e-16;
      d_z[i] = d_temp[i] / val;
    }
  }
};