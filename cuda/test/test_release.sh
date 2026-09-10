mkdir -p test_output
cd test_output
../build/release/nanoBraggCUDA -hkl ../test/P1.hkl -matrix ../test/A.mat -lambda 6.2 -N 10
cd ..