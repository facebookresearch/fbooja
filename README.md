The accompanying codes provide an implementation in PyTorch of the methods in
"Compressed sensing with a jackknife and a bootstrap" by Aaron Defazio, Mark
Tygert, Rachel Ward, and Jure Zbontar.

For help, try
python bootstrap.py --help
python bootstrap2.py --help
python jackknife.py --help
python jackknife2.py --help
python admm.py --help
python admm2.py --help

bootstrap.py and bootstrap2.py implement the bootstrap for estimating errors in
reconstructions from compressed sensing, the former for horizontal sampling,
and the latter for radial sampling.

jackknife.py and jackknife2.py implement the jackknife for estimating errors in
reconstructions from compressed sensing, the former for horizontal sampling,
and the latter for radial sampling.

admm.py and admm2.py perform compressed sensing via the alternating direction
method of multipliers for maximizing fidelity to Fourier-domain measurements
minus a total-variation penalty.

ctorch.py extends PyTorch to complex numbers.

radialines.py generates pixels on a Cartesian grid that intersect radial lines
emanating from the origin at specified angles.


Whereas bootstrap.py and jackknife.py should be useful for clinical practice,
bootstrap2.py and jackknife2.py are hacks full of code duplication from
bootstrap.py and jackknife.py, useful only for prototyping. The "2" versions
include admm2.py and radialines.py, which are similarly useful only for
experimentation. Radially retained sampling patterns used in clinical practice
do not fall on Cartesian grids. The "2" versions give a proof of principle,
nothing more.


********************************************************************************


brains.sh is a bash script that generates all figures in the paper available at
http://tygert.com/juq.pdf and http://tygert.com/juqs.pdf

brains.sh processes the scans of brains in the subdirectory "brains" and saves
the results to a new subdirectory, "brainsout".

Running brains.sh is required to process the LaTeX source in the subdirectory
"tex"; fbooja.tex and fboojas.tex depend both on each other and on the results
stored to the new subdirectory, "brainsout".


********************************************************************************


License

This fbooja software is licensed under the LICENSE file (the MIT license) in
the root directory of this source tree.
