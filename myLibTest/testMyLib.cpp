#include <iostream>

#include "../Base/TemplateFor.h"
#include "../linearAlgebra/vectorTemplate.h"

int main()
{
	std::cout << "Test base:" << std::endl;

	//Sum of natural numbers
	int s = 0;
	auto sum = [&s](int num)->void { s += num; };
	base::For<0, 11>::Do(sum);
	std::cout << "Check template for loop (sum from 0 to 10):" << s << std::endl;
	std::cout << "End of base testing.\n";

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
		<< "sum(v1 - v2) = " << math::sum(v1 - v2) << "\n";

    return 0;
}

