mkdir -p prof_output
cd prof_output
ncu --set full -f -o ../prof_output/nanoBraggCUDA ../build/release/nanoBraggCUDA -hkl ../test/P1.hkl -matrix ../test/A.mat -lambda 6.2 -N 10
cd ..
