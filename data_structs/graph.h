#ifndef GRAPH_H
#define GRAPH_H

#include <set>
#include <stack>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>

#include "../linearAlgebra/vectorTemplate.h"

/**
 * Unordered graph to keep mesh connectivity information
 */

namespace data_structs {

template<typename label>
class graph
{
	using edge_type = std::pair<label, label>;

    using label_list = std::set<label>;
    using adjacency_list = std::vector<label_list>;

    adjacency_list adjacency_list_;

public:

    /**
     * Creates an empty graph
     */
    graph(){}

    /**
     * Resets the connectivity array
     */
    void clear(){ adjacency_list_.clear(); }

    /**
     * Adds new connection into the unordered graph,
     * Note: node labels go in ascending order
     */
    void addEdge(label i, label j)
    {
        label max = std::max(i, j);
        label min = std::min(i, j);

        if(max >= adjacency_list_.size()) adjacency_list_.resize(max + 1);
		adjacency_list_[max].insert(min);
		adjacency_list_[min].insert(max);
    }

    /**
     * Get neighbour of the node
     */
    inline const label_list& getNeighbour(label node_label) const
    { return adjacency_list_[node_label]; }

    /**
     * Get number of nodes
     */
    inline size_t size() const { return adjacency_list_.size(); }

    /**
     * Returns a total number of connections
     */
    size_t connectionsNum() const
    {
        size_t connestions = 0;
        for(const label_list& ll : adjacency_list_)
            connestions += ll.size();
        return connestions/2;
    }

    /**
     * Iterates over unique connections passing corresponding labels into an observer functor
     */
    template<typename Observer>
    void iterateOverUniqueConnections(Observer observer) const
    {
        for(size_t i = 0; i != adjacency_list_.size(); ++i)
        {
            label_list::const_oterator 
				first = adjacency_list_[i].lower_bound(i),
				last  = adjacency_list_[i].end();
            for(; first != last; ++first) observer(i, *first);
        }
    }

    /**
     * Iterative deep first search from the start label
     */
    template<typename Obs, typename Pred>
	void dfs_iterative(label start, Pred pred, Obs obs) const
	{
		std::vector<bool> visited(adjacency_list_.size(), false);
		std::stack<label> dfs_stack;
		dfs_stack.push(start);

		while (!dfs_stack.empty())
		{
			label current_label = dfs_stack.top(); dfs_stack.pop();

			if (!visited[current_label] && pred(current_label))
			{
				obs(current_label);
				visited[current_label] = true;
				for (label l : getNeighbour(current_label)) dfs_stack.push(l);
			}
		}
	}
	/**
	 * Iterative breadth first search from the start label
	 */
	template<typename Obs>
	void bfs_iterative(label start, Obs obs) const
	{
		std::vector<bool> visited(adjacency_list_.size(), false);
		std::queue<label> bfs_queue;
		bfs_queue.push(start);

		while (!bfs_queue.empty())
		{
			label current_label = bfs_queue.front(); bfs_queue.pop();

			if (!visited[current_label])
			{
				visited[current_label] = true;
				if (obs(current_label))
					for (label l : getNeighbour(current_label)) bfs_queue.push(l);
			}
		}
	}
	/**
	 * Adds tetrahedra connectivity to a graph
	 */
	void addTet(const math::vector_c<label, 4>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[2]);
		addEdge(verts[0], verts[3]);
		addEdge(verts[1], verts[2]);
		addEdge(verts[1], verts[3]);
		addEdge(verts[2], verts[3]);
	}
	/**
	* Adds pyramid connectivity to a graph
	*/
	void addPyr(const math::vector_c<label, 5>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[3]);
		addEdge(verts[0], verts[4]);
		addEdge(verts[1], verts[2]);
		addEdge(verts[1], verts[4]);
		addEdge(verts[2], verts[3]);
		addEdge(verts[2], verts[4]);
		addEdge(verts[3], verts[4]);
	}
	/**
	* Adds wedge connectivity to a graph
	*/
	void addWedge(const math::vector_c<label, 6>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[2]);
		addEdge(verts[0], verts[3]);
		addEdge(verts[1], verts[2]);
		addEdge(verts[1], verts[4]);
		addEdge(verts[2], verts[5]);
		addEdge(verts[3], verts[4]);
		addEdge(verts[3], verts[5]);
		addEdge(verts[4], verts[5]);
	}
	/**
	 * Adds hexa connectivity
	 */
	void addHexa(const math::vector_c<label, 8>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[2]);
		addEdge(verts[0], verts[4]);
		addEdge(verts[1], verts[3]);
		addEdge(verts[1], verts[5]);
		addEdge(verts[2], verts[3]);
		addEdge(verts[2], verts[6]);
		addEdge(verts[3], verts[7]);
		addEdge(verts[4], verts[5]);
		addEdge(verts[4], verts[6]);
		addEdge(verts[5], verts[7]);
		addEdge(verts[6], verts[7]);
	}
	/**
	 * Adds triangle connectivity to a graph
	 */
	void addTri(const math::vector_c<label, 3>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[2]);
		addEdge(verts[1], verts[2]);
	}
	/**
	 *Adds square connectivity to a graph
	 */
	void addSq(const math::vector_c<label, 3>& verts)
	{
		addEdge(verts[0], verts[1]);
		addEdge(verts[0], verts[3]);
		addEdge(verts[1], verts[2]);
		addEdge(verts[2], verts[3]);
	}
};

}

#endif // GRAPH_H
