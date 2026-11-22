//                                MFEM Example 36 (AMR Version)
//
// Compile with: make ex36
//
// Description:  This is a modified version of Example 36 that includes:
//               1. Adaptive Mesh Refinement (AMR) loop using ZZ-Error Estimator.
//               2. Relative L1 Error evaluation against a Ground Truth mesh.
//               3. Obstacle re-projection from a dense mesh during adaptation.
//               4. Automatic feasibility repair after refinement.

#include <file_io/read_vtk.h>
#include <file_io/write_vtk.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <fstream> // Added for ofstream

#include "projector.h"
#include "coeff.h"
#include "error.h"
#include "mfem.hpp"
#include "preconditioner.h"

using namespace std;
using namespace mfem;

// ==========================================
// Helper Function: MFEM to Eigen
// ==========================================
void mfem_to_eigen(Mesh &mesh, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
  int nv = mesh.GetNV();
  int ne = mesh.GetNE();

  // Ensure V is always Nx3 for IGL compatibility
  V.resize(nv, 3);
  for (int i = 0; i < nv; i++)
  {
    const double *coords = mesh.GetVertex(i);
    if (mesh.SpaceDimension() == 2)
    {
      V(i, 0) = coords[0];
      V(i, 1) = coords[1];
      V(i, 2) = 0.0;
    }
    else
    {
      V(i, 0) = coords[0];
      V(i, 1) = coords[1];
      V(i, 2) = coords[2];
    }
  }

  // Handle Elements (Assuming Triangles for surface meshes)
  // For mixed meshes, this needs to be adapted.
  int n_verts = mesh.GetElement(0)->GetNVertices();
  F.resize(ne, n_verts);
  for (int i = 0; i < ne; i++)
  {
    Array<int> verts;
    mesh.GetElementVertices(i, verts);
    for (int j = 0; j < n_verts; ++j)
      F(i, j) = verts[j];
  }
}

// ==========================================
// Main Function
// ==========================================
int main(int argc, char *argv[])
{
  // 1. Parse command-line options
  const int order = 1;
  int max_newton_it = 100;
  int max_amr_it = 10;
  real_t alpha = 1.0;
  real_t tol = 1e-3;      // Global Error Tolerance (Rel L1)
  real_t step_tol = 1e-4; // Inner Solver Tolerance (Newton Step)
  string mesh_file, init_file, result_file, log_file, gt_file, obs_file;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
  args.AddOption(&obs_file, "-obs", "--obs", "Dense Obstacle Mesh file (defaults to GT if empty).");
  args.AddOption(&init_file, "-i", "--init", "Initial guess file.");
  args.AddOption(&result_file, "-r", "--result", "Result file prefix.");
  args.AddOption(&gt_file, "-gt", "--gt", "Ground truth file.");
  args.AddOption(&max_newton_it, "-mi", "--max-it", "Max inner solver iterations");
  args.AddOption(&max_amr_it, "-amr", "--max-amr", "Max AMR iterations");
  args.AddOption(&tol, "-tol", "--tol", "Global Stopping (Rel L1 Error)");
  args.AddOption(&step_tol, "-stol", "--step-tol", "Inner Stopping (Newton Step)");
  args.AddOption(&alpha, "-step", "--step", "Step size alpha");
  args.AddOption(&log_file, "-l", "--log", "Log file.");
  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }

  // 2. Setup Mesh & Data Loaders
  Mesh mesh(mesh_file, 1, 1);
  // mesh.EnsureNCMesh(false);
  int dim = mesh.Dimension();

  Eigen::VectorXd obs_vals;
  //* load sparse obstacle values *//
  {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    zxh::file_io::read_vtk_with_attribute(mesh_file, V, F, obs_vals);
  }

  Eigen::VectorXd gt_vals;

  // Load Ground Truth
  Eigen::MatrixXd V_gt;
  Eigen::MatrixXi F_gt;
  zxh::file_io::read_vtk_with_attribute(gt_file, V_gt, F_gt, gt_vals);
  ErrorEvaluator l1_error_evaluator(V_gt, F_gt, gt_vals);

  // Load Dense Obstacle
  Eigen::MatrixXd V_obs;
  Eigen::MatrixXi F_obs;
  Eigen::VectorXd obs_vals_gt;
  zxh::file_io::read_vtk_with_attribute(obs_file, V_obs, F_obs, obs_vals_gt);
  ObstacleProjector obs_projector(V_obs, F_obs, obs_vals_gt);

  Eigen::VectorXd init_vals;
  {
    Eigen::MatrixXd V_init;
    Eigen::MatrixXi F_init;
    zxh::file_io::read_vtk_with_attribute(init_file, V_init, F_init, init_vals);
  }

  // 3. Define Spaces
  H1_FECollection H1fec(order, dim);
  FiniteElementSpace H1fes(&mesh, &H1fec);

  L2_FECollection L2fec(order - 1, dim);
  FiniteElementSpace L2fes(&mesh, &L2fec);

  // 4. Initial Obstacle Projection
  GridFunction obstacle_gf(&H1fes);
  for (int i = 0; i < obstacle_gf.Size(); ++i)
    obstacle_gf(i) = obs_vals(i);
  GridFunctionCoefficient obstacle_coef(&obstacle_gf);

  // 5. Initialize Solution
  GridFunction u_gf(&H1fes);
  GridFunction psi_gf(&L2fes);
  for (int i = 0; i < u_gf.Size(); i++)
  {
    u_gf(i) = std::max(obstacle_gf(i), init_vals(i)) * 1.01;
  }

  ConstantCoefficient zero(0.0);
  LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle_coef);
  psi_gf.ProjectCoefficient(ln_u);

  YAML::Node config;
  const auto start_time = std::chrono::high_resolution_clock::now();
  int total_inner_iter = 0;
  real_t rel_l1_error = 1.0;

  // =========================================================
  // OUTER AMR LOOP
  // =========================================================
  int it_amr;
  for (it_amr = 0; it_amr < max_amr_it; ++it_amr)
  {
    spdlog::info(" AMR Iteration {} | DOFs: {}", it_amr, H1fes.GetVSize());

    // A. Update Evaluator with new mesh geometry
    Eigen::MatrixXd V_curr;
    Eigen::MatrixXi F_curr;
    mfem_to_eigen(mesh, V_curr, F_curr);
    l1_error_evaluator.update_geometry(V_curr, F_curr);

    // B. Setup Linear Algebra Objects
    //    We must rebuild these as the mesh size has changed
    Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = H1fes.GetVSize();
    offsets[2] = L2fes.GetVSize();
    offsets.PartialSum();

    BlockVector x(offsets), rhs(offsets);
    x = 0.0;
    rhs = 0.0;

    x.GetBlock(0) = u_gf;

    // Now link GridFunctions to the BlockVector
    u_gf.MakeRef(&H1fes, x, offsets[0]);
    GridFunction delta_psi_gf;
    delta_psi_gf.MakeRef(&L2fes, x, offsets[1]);
    delta_psi_gf = 0.0;

    GridFunction u_old_gf(&H1fes);
    GridFunction psi_old_gf(&L2fes);
    u_old_gf = u_gf;
    psi_old_gf = psi_gf;

    // C. Boundary Conditions
    Array<int> ess_bdr;
    if (mesh.bdr_attributes.Size())
    {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
    }

    // D. Assemble Constant Matrices (A00, A10)
    ConstantCoefficient alpha_cf(alpha);
    BilinearForm a00(&H1fes);
    a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
    a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_cf));
    a00.Assemble();
    a00.EliminateEssentialBC(ess_bdr, x.GetBlock(0), rhs.GetBlock(0), Operator::DIAG_ONE);
    a00.Finalize();
    SparseMatrix &A00 = a00.SpMat();

    MixedBilinearForm a10(&H1fes, &L2fes);
    a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
    a10.Assemble();
    a10.EliminateTrialEssentialBC(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
    a10.Finalize();
    SparseMatrix &A10 = a10.SpMat();
    SparseMatrix *A01 = Transpose(A10);

    // E. Setup A11 and Preconditioner
    BilinearForm a11(&L2fes);
    // A11 is updated inside the loop, we initialize it here
    ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
    ProductCoefficient neg_exp_psi(-1.0, exp_psi);
    a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
    a11.Assemble();
    a11.Finalize();
    SparseMatrix &A11 = a11.SpMat();

    // Precompute element volumes for fast diagonal update
    int ne = mesh.GetNE();
    Vector volumes(ne);
    for (int i = 0; i < ne; i++)
      volumes(i) = mesh.GetElementVolume(i);

    FixedEigenSolver prec_A00(A00);
    BlockTriangularPrec triangular_prec(prec_A00, A10, A11, offsets);

    BlockOperator A(offsets);
    A.SetBlock(0, 0, &A00);
    A.SetBlock(1, 0, &A10);
    A.SetBlock(0, 1, A01);
    A.SetBlock(1, 1, &A11);

    // 10. Iterate
    int k;
    int total_iterations = 0;
    real_t increment_u = 0.1;
    Eigen::VectorXd u_eigen = Eigen::VectorXd::Zero(mesh.GetNV());
    ConstantCoefficient f(0.0);
    for (k = 0; k < max_newton_it; k++)
    {
      GridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      int j;
      for (j = 0; j < 10; j++)
      {
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
          double *A11_data = A11.GetData();

          const double eps = 1e-6;
          const double max_exp = 1e6; // Cap for exp(psi) to prevent explosion

          // Parallel loop possible here
          for (int i = 0; i < ne; i++)
          {
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

        GMRES(A, triangular_prec, rhs, x, 0, 10000, 500, 1e-12, 0.0);

        u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
        delta_psi_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

        u_tmp -= u_gf;
        real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
        u_tmp = u_gf;

        real_t gamma = 1.0;
        delta_psi_gf *= gamma;
        psi_gf += delta_psi_gf;

        if (Newton_update_size < increment_u)
        {
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

      //* use error vs GT instead of increment error
      for (int i = 0; i < mesh.GetNV(); i++)
      {
        u_eigen(i) = u_gf(i);
      }

      rel_l1_error = l1_error_evaluator.compute_rel_l1(u_eigen);
      config["rel_l1_error"].push_back(rel_l1_error);
      if (rel_l1_error < tol || k == max_newton_it - 1)
      {
        break;
      }
    }

    // 1. Check Global Convergence
    if (rel_l1_error < tol)
    {
      spdlog::info("Converged: Rel L1 Error {} < Tolerance {}", rel_l1_error, tol);
      break;
    }

    // 2. Check Max AMR Iterations
    if (it_amr >= max_amr_it - 1)
    {
      spdlog::info("Reached maximum AMR iterations ({}). Stopping.", max_amr_it);
      break;
    }

    // 3. Adapt Mesh

    // =====================================================
    // Adapt Mesh
    // =====================================================

    // 1. Allocate Integrator (We must delete this manually later)
    DiffusionIntegrator *estimator_integ = new DiffusionIntegrator(alpha_cf);

    // 2. Allocate Flux Space on HEAP
    // The ZienkiewiczZhuEstimator takes ownership and will delete this automatically.
    FiniteElementSpace *flux_fespace = new FiniteElementSpace(&mesh, &H1fec, dim);

    // 3. Setup Estimator
    // We pass *estimator_integ (reference) and flux_fespace (pointer)
    ZienkiewiczZhuEstimator estimator(*estimator_integ, u_gf, flux_fespace);
    estimator.SetAnisotropic();

    // 4. Setup Refiner
    ThresholdRefiner refiner(estimator);
    refiner.SetTotalErrorFraction(0.7);

    // 5. Apply Refinement
    refiner.Apply(mesh);

    // 6. Clean up the Integrator (Estimator does NOT own the integrator)
    delete estimator_integ;
    // Note: flux_fespace is deleted automatically when 'estimator' goes out of scope

    if (refiner.Stop())
    {
      spdlog::info("No elements marked for refinement. Stopping AMR.");
      break;
    }

    // 4. Update Spaces and Fields
    H1fes.Update();
    L2fes.Update();

    // Interpolate Solution (Standard MFEM interpolation)
    u_gf.Update();
    psi_gf.Update();
    obstacle_gf.Update();

    // 5. CRITICAL: Re-project Obstacle from High-Res Source
    //    Standard Update() just linearly interpolates the old coarse obstacle.
    //    We need the sharp features from the original dense mesh.
    Eigen::MatrixXd V_new;
    Eigen::MatrixXi F_new;
    mfem_to_eigen(mesh, V_new, F_new);

    Eigen::VectorXd new_obs_vals = obs_projector.query(V_new);

    // Update Obstacle GridFunction
    for (int i = 0; i < obstacle_gf.Size(); ++i)
    {
      obstacle_gf(i) = new_obs_vals(i);
    }

    // 6. SAFETY REPAIR
    //    The new sharp obstacle might poke through the interpolated u_gf.
    //    We must ensure u >= obs + margin to keep log(u-obs) valid.
    int repair_count = 0;
    const double margin = 1e-6;
    for (int i = 0; i < u_gf.Size(); ++i)
    {
      if (u_gf(i) <= obstacle_gf(i) + margin)
      {
        u_gf(i) = obstacle_gf(i) + margin;
        repair_count++;
      }
    }
    if (repair_count > 0)
    {
      spdlog::warn("Repaired {} DOFs to maintain feasibility after adaptation.", repair_count);
    }

    // 7. Reset Psi
    //    Since u and obs changed discontinuously, recompute psi from definition.
    LogarithmGridFunctionCoefficient new_ln_u(u_gf, obstacle_coef);
    psi_gf.ProjectCoefficient(new_ln_u);
  }

  // ==========================================
  // Final Output
  // ==========================================
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;

  // Log stats
  config["total_time"] = elapsed_seconds.count();
  config["final_rel_l1"] = rel_l1_error;
  config["DOF"] = mesh.GetNV();
  config["total_adaptations"] = it_amr + 1;
  if (!log_file.empty())
  {
    std::ofstream fout(log_file);
    fout << config;
  }

  // Save Final Solution
  // Need to extract new F/V for saving
  Eigen::MatrixXd V_final;
  Eigen::MatrixXi F_final;
  mfem_to_eigen(mesh, V_final, F_final);

  Eigen::VectorXd u_final(mesh.GetNV());
  for (int i = 0; i < mesh.GetNV(); i++)
    u_final(i) = u_gf(i);

  zxh::file_io::write_vtk_with_attribute(result_file, V_final, F_final, u_final, "displacement");

  return 0;
}