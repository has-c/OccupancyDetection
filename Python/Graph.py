class Graph(object):

    def __init__(self, graphDict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graphDict == None:
            graphDict = {}
        self.graphDict = graphDict

    def getVertices(self):
        """ returns the vertices of a graph """
        return list(self.graphDict.keys())

    def getEdges(self):
        """ returns the edges of a graph """
        return self.generateEdges()

    def addVertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.graphDict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.graphDict:
            self.graphDict[vertex] = []

    def addEdge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.graphDict:
            self.graphDict[vertex1].append(vertex2)
        else:
            self.graphDict[vertex1] = [vertex2]

    def generateEdges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self.graphDict:
            for neighbour in self.graphDict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.graphDict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.generateEdges():
            res += str(edge) + " "
        return res