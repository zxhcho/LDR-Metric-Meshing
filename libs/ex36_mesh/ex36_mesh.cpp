//                                MFEM Example 36
//
// Compile with: make ex36
//
// Sample runs: ex36 -o 2
//              ex36 -o 2 -r 4
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ ϕ in H¹₀.
//
//              This is known as the obstacle problem, and it is a simple
//              mathematical model for contact mechanics.
//
//              In this example, the obstacle ϕ is a half-sphere centered
//              at the origin of a circular domain Ω. After solving to a
//              specified tolerance, the numerical solution is compared to
//              a closed-form exact solution to assess accuracy.
//
//              The problem is discretized and solved using the proximal
//              Galerkin finite element method, introduced by Keith and
//              Surowiec [1].
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to variation inequality problems and
//              showcases how to set up and solve nonlinear mixed methods.
//
// [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]

#include <file_io/read_vtk.h>
#include <file_io/write_vtk.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <chrono>
#include <iostream>

#include "coeff.h"
#include "error.h"
#include "mfem.hpp"
#include "preconditioner.h"

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // 1. Parse command-line options.
  const int order = 1;
  int max_it = 10;
  real_t alpha = 1.0;
  real_t tol = 1e-5;
  real_t step_tol = 1e-4;
  string mesh_file, init_file, result_file, log_file, gt_file;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file storing the domain and obstacle values.");
  args.AddOption(&init_file, "-i", "--init",
                 "Initial guess file storing initial solution values.");
  args.AddOption(&result_file, "-r", "--result", "Mesh file with solution.");
  args.AddOption(&gt_file, "-gt", "--gt",
                 "Mesh file with ground truth solution.");
  args.AddOption(&max_it, "-mi", "--max-it", "Maximum number of iterations");
  args.AddOption(&tol, "-tol", "--tol",
                 "Stopping criteria based on the difference between"
                 "successive solution updates");
  args.AddOption(&step_tol, "-stol", "--step-tol",
                 "Inner Stopping (Newton Step)");
  args.AddOption(&alpha, "-step", "--step", "Step size alpha");
  args.AddOption(&log_file, "-l", "--log", "Log file to record output.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }

  // 2. Read the calculation domain from the mesh file.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  Eigen::MatrixXd V, V_gt;
  Eigen::MatrixXi F, F_gt;
  Eigen::VectorXd obstacle_values, initial_values, gt_values;
  zxh::file_io::read_vtk_with_attribute(mesh_file, V, F, obstacle_values);
  zxh::file_io::read_vtk_with_attribute(gt_file, V_gt, F_gt, gt_values);

  const bool is_init_exist = init_file.empty() ? false : true;
  if (is_init_exist) {
    zxh::file_io::read_vtk_with_attribute(init_file, V, F, initial_values);
  }

  ErrorEvaluator evaluator(V_gt, F_gt, gt_values);
  evaluator.update_geometry(V, F);

  const int nv = mesh.GetNV();
  const int ne = mesh.GetNE();
  Eigen::VectorXd u_eigen = Eigen::VectorXd::Zero(nv);

  YAML::Node config;
  // 3. Start the timer. exclusing the file loading time
  const auto start_time = std::chrono::high_resolution_clock::now();

  // 4. Define the necessary finite element spaces on the mesh.
  H1_FECollection H1fec(order, dim);
  FiniteElementSpace H1fes(&mesh, &H1fec);

  L2_FECollection L2fec(order - 1, dim);
  FiniteElementSpace L2fes(&mesh, &L2fec);

  GridFunction obstacle_gf(&H1fes);
  for (int i = 0; i < H1fes.GetVSize(); i++) {
    obstacle_gf(i) = obstacle_values(i);
  }
  GridFunctionCoefficient obstacle_coef(&obstacle_gf);

  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = H1fes.GetVSize();
  offsets[2] = L2fes.GetVSize();
  offsets.PartialSum();

  BlockVector x(offsets), rhs(offsets);
  x = 0.0;
  rhs = 0.0;

  // 6. Define an initial guess for the solution.
  ConstantCoefficient zero(0.0);

  // 7. Define the solution vectors as a finite element grid functions
  //    corresponding to the fespaces.
  GridFunction u_gf, delta_psi_gf;

  u_gf.MakeRef(&H1fes, x, offsets[0]);
  delta_psi_gf.MakeRef(&L2fes, x, offsets[1]);
  delta_psi_gf = 0.0;

  if (initial_values.size() == u_gf.Size()) {
    for (int i = 0; i < u_gf.Size(); i++) {
      u_gf(i) = std::max(obstacle_gf(i), initial_values(i)) * 1.01;
    }
  } else {
    for (int i = 0; i < u_gf.Size(); i++) {
      u_gf(i) = obstacle_gf(i) * 1.01;
    }
  }

  // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
  Array<int> ess_bdr;
  if (mesh.bdr_attributes.Size()) {
    ess_bdr.SetSize(mesh.bdr_attributes.Max());
    ess_bdr = 1;
  }

  GridFunction u_old_gf(&H1fes);
  GridFunction psi_old_gf(&L2fes);
  GridFunction psi_gf(&L2fes);
  u_old_gf = 0.0;
  psi_old_gf = 0.0;

  // 8. Define the function coefficients for the solution and use them to
  //    initialize the initial guess
  ConstantCoefficient f(0.0);
  u_old_gf = u_gf;

  // 9. Initialize the slack variable ψₕ = ln(uₕ)
  LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle_coef);
  psi_gf.ProjectCoefficient(ln_u);
  psi_old_gf = psi_gf;

  // construct A
  ConstantCoefficient alpha_cf(alpha);
  BilinearForm a00(&H1fes);
  a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
  a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_cf));
  a00.Assemble();
  a00.EliminateEssentialBC(ess_bdr, x.GetBlock(0), rhs.GetBlock(0),
                           mfem::Operator::DIAG_ONE);
  a00.Finalize();
  SparseMatrix& A00 = a00.SpMat();

  MixedBilinearForm a10(&H1fes, &L2fes);
  a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
  a10.Assemble();
  a10.EliminateTrialEssentialBC(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
  a10.Finalize();
  SparseMatrix& A10 = a10.SpMat();
  SparseMatrix* A01 = Transpose(A10);

  BilinearForm a11(&L2fes);
  ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
  ProductCoefficient neg_exp_psi(-1.0, exp_psi);
  a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
  a11.Assemble();
  a11.Finalize();
  SparseMatrix& A11 = a11.SpMat();

  // Pre-calculate Element Volumes (for fast update)
  Vector volumes(ne);
  for (int i = 0; i < ne; i++) {
    volumes(i) = mesh.GetElementVolume(i);
  }

  // We create the smoothers manually to prevent re-setup.
  // GSSmoother prec_A00(A00); // Constant structure -> Setup once!
  FixedEigenSolver prec_A00(A00);

  BlockTriangularPrec triangular_prec(prec_A00, A10, A11, offsets);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, &A00);
  A.SetBlock(1, 0, &A10);
  A.SetBlock(0, 1, A01);
  A.SetBlock(1, 1, &A11);  // Points to the same matrix object

  // 10. Iterate
  int k;
  int total_iterations = 0;
  real_t increment_u = 0.1;
  for (k = 0; k < max_it; k++) {
    GridFunction u_tmp(&H1fes);
    u_tmp = u_old_gf;

    int j;
    for (j = 0; j < 10; j++) {
      total_iterations++;

      LinearForm b0, b1;
      b0.Update(&H1fes, rhs.GetBlock(0), 0);
      b1.Update(&L2fes, rhs.GetBlock(1), 0);

      ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
      ProductCoefficient neg_exp_psi(-1.0, exp_psi);
      ProductCoefficient alpha_f(alpha, f);
      GridFunctionCoefficient psi_cf(&psi_gf);
      GridFunctionCoefficient psi_old_cf(&psi_old_gf);
      SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

      b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
      b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
      b0.Assemble();

      b1.AddDomainIntegrator(new DomainLFIntegrator(exp_psi));
      b1.AddDomainIntegrator(new DomainLFIntegrator(obstacle_coef));
      b1.Assemble();

      // --- STEP 1: FAST UPDATE of A11 (The "Update NNZ" Strategy) ---
      // A11 = Mass(-exp(psi) - epsilon)
      // For P0, this is Diagonal. Value[i] = (-exp(psi_i) - eps) * Volume_i
      {
        // Access raw data array of SparseMatrix (CSR format).
        // For Diagonal matrix, Data[i] is simply the diagonal entry (i,i).
        double* A11_data = A11.GetData();

        const double eps = 1e-6;
        const double max_exp = 1e6;  // Cap for exp(psi) to prevent explosion

        // Parallel loop possible here
        for (int i = 0; i < ne; i++) {
          double psi_val = psi_gf(i);

          // 1. Compute Exp with Clamping
          // If psi > 13.8, exp(psi) > 1e6. We clamp it.
          double exp_val;
          if (psi_val > 14.0)
            exp_val = max_exp;
          else
            exp_val = exp(psi_val);

          // 2. Apply Weight: -exp(psi) - epsilon
          double weight = -exp_val - eps;

          // 3. Update Matrix Diagonal
          A11_data[i] = weight * volumes(i);
        }
      }

      // Since we modified values of A11, we must notify the Smoother
      // (Depends on implementation, GSSmoother usually grabs diagonal on Setup)
      // To be safe and fast:
      // prec_A11.SetOperator(A11); // Cheap re-link

      GMRES(A, triangular_prec, rhs, x, 0, 10000, 500, 1e-12, 0.0);

      u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
      delta_psi_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

      u_tmp -= u_gf;
      real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
      u_tmp = u_gf;

      real_t gamma = 1.0;
      delta_psi_gf *= gamma;
      psi_gf += delta_psi_gf;

      if (Newton_update_size < increment_u) {
        break;
      }
    }

    u_tmp = u_gf;
    u_tmp -= u_old_gf;
    increment_u = u_tmp.ComputeL2Error(zero);

    u_old_gf = u_gf;
    psi_old_gf = psi_gf;

    auto end_time_iter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_iter =
        end_time_iter - start_time;
    config["time"].push_back(elapsed_seconds_iter.count());
    for (int i = 0; i < nv; i++) {
      u_eigen(i) = u_gf(i);
    }

    //* use error vs GT instead of increment error
    const double rel_l1_err = evaluator.compute_rel_l1(u_eigen);
    config["rel_l1_error"].push_back(rel_l1_err);
    if (u_tmp.ComputeL1Error(zero) < step_tol || rel_l1_err < tol ||
        k == max_it - 1) {
      break;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;

  if (log_file.empty()) {
    spdlog::info(
        "Time elapsed for {} outer iterations with {} inner iterations: {}s ",
        k + 1, total_iterations, elapsed_seconds.count());
  } else {
    config["outer_iter_total"] = k + 1;
    config["inner_iter_total"] = total_iterations;
    config["total_time"] = elapsed_seconds.count();
    const double rel_l1_err = evaluator.compute_rel_l1(u_eigen);
    config["final_rel_l1"] = rel_l1_err;
    config["DOF"] = nv;
    std::ofstream fout(log_file);
    fout << config;
  }

  for (int i = 0; i < nv; i++) {
    u_eigen(i) = u_gf(i);
  }
  zxh::file_io::write_vtk_with_attribute(result_file, V, F, u_eigen,
                                         "displacement");

  return 0;
}
