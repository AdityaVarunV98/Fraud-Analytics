"""pagerank.py illustrates how to use the pregel.py library, and tests
that the library works.

It illustrates pregel.py by computing the PageRank for a randomly
chosen 10-vertex web graph.

It tests pregel.py by computing the PageRank for the same graph in a
different, more conventional way, and showing that the two outputs are
near-identical."""

from pregel import Vertex, Pregel

# The next two imports are only needed for the test.  
from numpy import asmatrix, eye, zeros, ones, linalg
import random, csv
import pandas as pd

num_workers = 4
Vertices = {}

def main():

    totaltrust = make_edges()
    vertexlist = list(Vertices.values())
    print(totaltrust)
    for u in vertexlist:
        u.value = u.value / totaltrust
        u.initial_value = u.value
        print(u.value)
    print("done")
    
    pr_test = trustrank_test(vertexlist)
    print("Test computation of pagerank:\n%s" % pr_test)
    pr_pregel = trustrank_pregel(vertexlist)
    print("Pregel computation of pagerank:\n%s" % pr_pregel)
    diff = pr_pregel-pr_test
    print("Difference between the two pagerank vectors:\n%s" % diff)
    print("The norm of the difference is: %s" % linalg.norm(diff))

def make_edges():
    df_transactions = pd.read_csv("Data/Payments.csv")

    # Convert to edges list
    edges = list(df_transactions.itertuples(index=False, name=None))

    df = pd.DataFrame(edges, columns=['sender', 'receiver', 'amount'])
    total_sent = df.groupby('sender')['amount'].sum().rename('total_sent')
    df = df.merge(total_sent, on='sender')
    df['weight'] = df['amount'] / df['total_sent']
    df_grouped = df.groupby(['sender', 'receiver'])['weight'].sum().reset_index()

    df_badsenders = pd.read_csv("Data/bad_sender.csv")
    fraud_ids = set(df_badsenders['Bad Sender'])

    print (df_grouped.size)
    totaltrust = 0
    for u, v, w in df_grouped.itertuples(index=False, name=None):
        # check if u and v exist     
        if u not in Vertices.keys():
            trust = 1
            if u in fraud_ids:
                trust = 0
            Vertices[u] = TrustRankVertex(u, trust, [])
            totaltrust += trust
        if v not in Vertices.keys():
            trust = 1
            if v in fraud_ids:
                trust = 0
            Vertices[v] = TrustRankVertex(v, trust, [])
            totaltrust += trust
        Vertices[u].out_vertices.append((Vertices[v], w))

    return totaltrust



def trustrank_test(vertices):
    """Computes the pagerank vector associated to vertices, using a
    standard matrix-theoretic approach to computing pagerank.  This is
    used as a basis for comparison."""
    I = asmatrix(eye(len(vertices)))
    G = zeros((len(vertices), len(vertices)))
    mp = {}
    index = 0
    for vertex in vertices:
        mp[vertex.id] = index
        index += 1 
    for vertex in vertices:
        for out_vertex, weight in vertex.out_vertices:
            print(vertex.id, out_vertex.id, weight)
            G[mp[out_vertex.id], mp[vertex.id]] = weight

    P = [v.value for v in vertices]
    P = asmatrix(P)
    P = P.T
    # P = (1.0/num_vertices)*asmatrix(ones((num_vertices,1)))
    return 0.15*((I-0.85*G).I)*P

def trustrank_pregel(vertices):
    """Computes the pagerank vector associated to vertices, using
    Pregel."""
    p = Pregel(vertices, num_workers)
    p.run()
    return asmatrix([vertex.value for vertex in p.vertices]).transpose()

class TrustRankVertex(Vertex):

    def __init__(self, id, value, out_vertices):
        super().__init__(id, value, out_vertices)  
        self.initial_value = value 

    def update(self):
        # This routine has a bug when there are pages with no outgoing
        # links (never the case for our tests).  This problem can be
        # solved by introducing Aggregators into the Pregel framework,
        # but as an initial demonstration this works fine.
        if self.superstep < 50:
            self.value = 0.15 * self.initial_value + 0.85 * sum([trustrank for (vertex, trustrank) in self.incoming_messages])           
            self.outgoing_messages = [(vertex, self.value * weight) for (vertex, weight) in self.out_vertices]
        else:
            self.active = False

if __name__ == "__main__":
    main()
