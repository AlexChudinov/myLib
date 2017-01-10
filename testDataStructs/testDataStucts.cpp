#include "../data_structs/graph.h"
#include <iostream>
#include <cstddef>

int main()
{
	std::cout << "Test graph:" << std::endl;
	data_structs::graph<uint32_t> g;

	//Create graph
	g.addEdge(0, 1);
	g.addEdge(1, 2);
	g.addEdge(2, 3);
	g.addEdge(3, 0);
	//And print all labels along the search direction
	g.dfs_iterative(0, [](uint32_t)->bool { return true; }, [&](uint32_t l) { std::cout << l << "\n"; });
}