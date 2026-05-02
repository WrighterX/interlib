.PHONY: matlab-build matlab-session matlab-test matlab-test-batch matlab-test-installation matlab-test-installation-batch matlab-test-newton matlab-test-newton-batch matlab-test-quadratic matlab-test-quadratic-batch matlab-test-cubic-spline matlab-test-cubic-spline-batch matlab-test-rbf matlab-test-rbf-batch matlab-test-chebyshev matlab-test-chebyshev-batch matlab-toolbox-stage matlab-toolbox-build matlab-toolbox-package matlab-toolbox-package-batch

matlab-build:
	CARGO_TARGET_DIR="$(CURDIR)/target/matlab" cargo build --lib --no-default-features --features matlab

matlab-toolbox-stage: matlab-build
	bash ./scripts/stage_matlab_toolbox.sh

matlab-toolbox-build: matlab-toolbox-stage
	bash ./scripts/build_matlab_toolbox.sh

matlab-toolbox-package: matlab-toolbox-build
	bash ./scripts/package_matlab_toolbox.sh

matlab-toolbox-package-batch: matlab-toolbox-build
	MATLAB_BATCH=1 MATLAB_IMAGE=$${MATLAB_IMAGE:-my-matlab-image:auth} bash ./scripts/package_matlab_toolbox.sh

matlab-session:
	bash ./scripts/start_matlab_container.sh

matlab-test: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_linear_test.sh

matlab-test-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_linear_test.sh

matlab-test-installation: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_installation_test.sh

matlab-test-installation-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_installation_test.sh

matlab-test-newton: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_newton_test.sh

matlab-test-newton-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_newton_test.sh

matlab-test-hermite: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_hermite_test.sh

matlab-test-hermite-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_hermite_test.sh

matlab-test-lagrange: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_lagrange_test.sh

matlab-test-lagrange-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_lagrange_test.sh

matlab-test-least-squares: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_least_squares_test.sh

matlab-test-least-squares-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_least_squares_test.sh

matlab-test-rbf: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_rbf_test.sh

matlab-test-rbf-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_rbf_test.sh

matlab-test-chebyshev: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_chebyshev_test.sh

matlab-test-chebyshev-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_chebyshev_test.sh

matlab-test-quadratic: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_quadratic_test.sh

matlab-test-quadratic-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_quadratic_test.sh

matlab-test-cubic-spline: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_cubic_spline_test.sh

matlab-test-cubic-spline-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_cubic_spline_test.sh
