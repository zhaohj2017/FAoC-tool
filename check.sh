#！/bin/bash

# script for syntheiszing controller and verifying it

# train nn model save in redlog format
python3 main.py

# generate vefication condition by redlog for isat3
redcsl --nogui < isat.rl

# delete empty lines
sed -i ":a;N;s/\n//g;ta" init_veri.hy
sed -i ":a;N;s/\n//g;ta" safe_veri.hy
sed -i ":a;N;s/\n//g;ta" lie_veri.hy

# power: ** --> ^
sed -i "s/\*\*/\^/g" init_veri.hy
sed -i "s/\*\*/\^/g" safe_veri.hy
sed -i "s/\*\*/\^/g" lie_veri.hy

# replace dollar by ; and add new lines
sed -i 's/\$/;\n/g' init_veri.hy
sed -i 's/\$/;\n/g' safe_veri.hy
sed -i 's/\$/;\n/g' lie_veri.hy

# remove superfluous ;
sed -i 's/DECL;/DECL/g' init_veri.hy
sed -i 's/EXPR;/EXPR/g' init_veri.hy
sed -i 's/DECL;/DECL/g' safe_veri.hy
sed -i 's/EXPR;/EXPR/g' safe_veri.hy
sed -i 's/DECL;/DECL/g' lie_veri.hy
sed -i 's/EXPR;/EXPR/g' lie_veri.hy

# call isat3
echo "=========================================================="
echo "Checking init:"
time isat3 -I --msw 0.001 -v init_veri.hy | grep "UNSATISFIABLE" 
echo "=========================================================="
echo "Checking unsafe:"
time isat3 -I --msw 0.001 -v safe_veri.hy | grep "UNSATISFIABLE" 
echo "=========================================================="
echo "Checking Lie:"
time isat3 -I --msw 0.001 -v lie_veri.hy | grep "UNSATISFIABLE"
echo "=========================================================="


## delete temp files
rm init_veri.hy
rm safe_veri.hy
rm lie_veri.hy
rm nnredlog.txt
