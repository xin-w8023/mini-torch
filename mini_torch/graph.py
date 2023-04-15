import queue

from mini_torch.tensor import Tensor


def to_graph(tensor):
    default_style = {"shape": "rectangle"}
    act_style = {"shape": "egg", "color": "red"}
    leaf_style = {"shape": "rectangle", "color": "lightblue", "style": "filled"}
    op_style = {"shape": "oval"}
    import graphviz

    visited = {}
    edges_set = set()
    dot = graphviz.Digraph(name="Graph", graph_attr=dict(size="800,160"))
    dot.attr(newrank="true")
    q = queue.Queue()
    global_i = -1
    dummy_node = Tensor(1)
    dummy_node.pre.append(tensor)
    nodes, edges = [], []

    q.put((global_i, dummy_node))
    while not q.empty():
        index, node = q.get()
        for n in node.pre:
            if n not in visited:
                style = default_style
                if not n.pre:
                    style = leaf_style
                if not n.next:
                    style = act_style

                if n.grad_fn in ["+", "-", "*", "/"]:
                    style = op_style

                global_i += 1
                nodes.append(
                    dict(
                        name=str(global_i),
                        label=f"{n.data}|{n.grad}",
                        _attributes=style,
                    )
                )
                q.put((global_i, n))
                visited[n] = str(global_i)

            if (visited[n], str(index)) not in edges and index != -1:
                edges.append(
                    dict(tail_name=visited[n], head_name=str(index), constraint="false")
                )
                edges_set.add((visited[n], str(index)))
    for node in nodes[::-1]:
        dot.node(**node)
    for edge in edges:
        dot.edge(**edge)
    return dot


def show(tensor):
    to_graph(tensor).render(view=True, cleanup=True)
