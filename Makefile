.PHONY: matlab-build matlab-session matlab-test matlab-test-batch matlab-test-newton matlab-test-newton-batch matlab-test-quadratic matlab-test-quadratic-batch matlab-test-cubic-spline matlab-test-cubic-spline-batch

matlab-build:
	CARGO_TARGET_DIR="$(CURDIR)/target/matlab" cargo build --lib --no-default-features --features ffi

matlab-session:
	bash ./scripts/start_matlab_container.sh

matlab-test: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_linear_test.sh

matlab-test-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_linear_test.sh

matlab-test-newton: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_newton_test.sh

matlab-test-newton-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_newton_test.sh

matlab-test-quadratic: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_quadratic_test.sh

matlab-test-quadratic-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_quadratic_test.sh

matlab-test-cubic-spline: matlab-build
	MATLAB_CONTAINER=$${MATLAB_CONTAINER:-matlab-login} bash ./scripts/run_matlab_cubic_spline_test.sh

matlab-test-cubic-spline-batch: matlab-build
	MATLAB_BATCH=1 bash ./scripts/run_matlab_cubic_spline_test.sh
