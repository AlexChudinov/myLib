#include <iostream>
#include <fstream>
#include <iterator>
#include <ctime>

#include "../Base/templateFor.h"
#include "../linearAlgebra/vectorTemplate.h"
#include "../linearAlgebra/matrixTemplate.h"

#include "../Interpolation/InterpFun.h"

int main()
{
	std::cout << "Test base:" << std::endl;

	//Sum of natural numbers
	int s = 0;
	auto sum = [&s](int num)->void { s += num; };
	base::For<0, 11>::Do(sum);
	std::cout << "Check template for loop (sum from 0 to 10) = " << s << std::endl;
	std::cout << "End of base testing.\n\n";

	std::cout << "Test vector template: " << std::endl;
	using v3d = math::vector_c<double, 3>;
	v3d
		v{ 1.,2.,3. },
		v1{ 4.,5.,6. },
		v2 = v;
	std::cout << "v1 = " << v1 << "\n";
	std::cout << "v2 = " << v2 << "\n";
	std::cout << "(v1-=10)*(v2+=3) = " << ((v1-=10.)*(v2+=3.)) << "\n";
	std::cout << "v1 = " << v1 << "\n"
		<< "v1 * 2 = " << (v1 * 2.) << " = " << (2. * v1) << "\n"
		<< "- v1 = " << (-v1) << "\n"
		<< "abs(v1/2.) = " << math::abs(v1 / 2.) << "\n"
		<< "prod(v1) = " << math::prod(v1) << "\n"
		<< "sum(v1 + v2) = " << math::sum(v1 + v2) << "\n"
		<< "sum(v1 - v2) = " << math::sum(v1 - v2) << "\n"
		<< "End of vector template testing.\n\n";

	std::cout << "Test matrix template" << std::endl;
	using m3x3 = math::matrix_c<double, 3, 3>;
	m3x3 m1{ {1., 2., 3.}, {4., 5., 6.}, {7., 8., 19.} };
	m3x3 m2 = math::eye<double, 3>();
	std::cout << "m1 = \n" << m1 << "\n";
	std::cout << "m1 + m2 = \n" << (m1 + m2) << "\n";
	std::cout << "m1 - (-m2) = \n" << (m1 - (-m2)) << "\n";
	std::cout << "v1*m1 = " << (v1 * m1) << "\n";
	std::cout << "m1*v1 = " << (m1 * v1) << "\n";
	std::cout << "m1*m1 = \n" << (m1 * m1) << "\n";
	std::cout << "m1/10 = \n" << (m1 / 10.) << "\n 3*(m1*10) = \n" << (3.*(m1*10.)) << "\n";
	//Compare perfomance with matlab
	std::clock_t start = std::clock();
	const int nMults = 1000;
	for (int j = 0; j < nMults; ++j) math::qrGramSchmidt(m1);
	std::cout << "Time for " << nMults << " multiplications is " << (std::clock() - start) / CLOCKS_PER_SEC << " seconds.\n";
	std::cout << "transpose(m1) =\n" << math::transpose(m1) << "\n";
	std::cout << "cov = \n"
		<< math::cov(std::vector<v3d>{ {1., 2., 3.}, { 4., 5., 6. }, { 7., 8., 9. } }) << "\n"
		<< math::cov(std::vector<v3d>{ {1., 2., 3.}, { 4., 5., 6. }, { 7., 8., 9. } }, { 1.,1.,1. }) << "\n"
		<< math::qrGramSchmidt(m1).first << "\n"
		<< math::qrGramSchmidt(m1).second << "\n"
		<< math::qrGramSchmidt(m1).first * math::qrGramSchmidt(m1).second << "\n"
		<< "tr(m1) = " << math::tr(m1) << "\n"
		<< "eigvals(m1) = " << math::eigenValsQRGramSchmidt(m1) << "\n"
		<< "eigvect(m1) = " << math::eigenVectorSimple(m1) << "\n";
	//Try calculate next eigen vectors
	std::cout << "eigen vector = " << math::eigenVectorsSimple(m1) << "\n";

	std::cout << "det_gaus(m1) = " << math::det_gaus(m1) << std::endl;
	std::cout << "det(m1) = " << math::det(m1) << std::endl;

	m3x3 cov;
	std::vector<v3d> vects{ { 2., 2., 3. },{ 2.,0.,3. },{ 3.,3.,4. },{ 4., -2., 5. } };
	std::cout << "\ncov = " << (cov = math::cov(vects)) 
		<< "\n";
	v3d e1, e2;
	std::cout << "e1 = " << (e1 = math::eigenVectorSimple(cov)) << "\n";
	std::for_each(vects.begin(), vects.end(), [&](v3d& vec) { vec = vec - (vec*e1)*e1; });
	v3d n = math::crossProduct(e1, e2 = math::eigenVectorSimple(math::cov(vects)));
	std::cout << "n = " << n << "\n";

	//Spline test
	InterpFun<double>::PInterpFun pFun = InterpFun<double>::create(
		InterpFun<double>::ECubic,
		InterpFun<double>::XYDataVector{{1,1},{2,3},{3,9},{4,16},{5,25}}
	);
	
	std::vector<double> x(100), y(100);
	y[0] = pFun->interpolate(0.0);
	for (size_t i = 1; i < 100; ++i) {
		x[i] = x[i - 1] + .05;
		y[i] = pFun->interpolate(x[i]);
	}

	std::ofstream out;
	out.open("test_out.txt");
	std::copy(y.begin(), y.end(), std::ostream_iterator<double>(out, "\n"));
	out.close();

    return 0;
}

