"""
cluster.py
"""
import pprint, pickle
import networkx as nx
import matplotlib.pyplot as plt

def read_friend_and_follower_IDs():
    pkl_file_friends_and_follower_IDs = open('Collected_Data/friends_and_follower_IDs.pkl','rb')
    return pickle.load(pkl_file_friends_and_follower_IDs)
    
def create_graph():
    return nx.Graph()
     
def add_users_to_graph(graph, friends_and_follower_IDs):
    top_user_screen_names = [screen_name for screen_name in friends_and_follower_IDs.keys()]
    for screen_name in top_user_screen_names:
        graph.add_node(screen_name)
        completeList = friends_and_follower_IDs[screen_name]
        for eachPerson in completeList:
            graph.add_edge(screen_name, eachPerson)
    draw_graph(graph)
    return graph    

def girvan_newman(G, depth=0):

    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)

        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        #print(indent + 'removing ' + str(edge_to_remove))
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c.nodes() for c in components]
    #print(indent + 'components=' + str(result))
    for c in components:
        result.extend(girvan_newman(c, depth + 1))

    return result



def draw_graph(graph):
    nx.draw_networkx(graph, layout =nx.spring_layout(graph), node_size = 40)
    plt.savefig("Cluster_Data/graph.png", format = "PNG")
    plt.tight_layout()
    plt.show()

def main():
    # Fetch data from .pkl files
    # Use: friends_and_follower_IDs.pkl 
    
    friends_and_follower_IDs = read_friend_and_follower_IDs()
    print("Now, we have a dictionary. Screen Name to Friend and Follower IDs.")
    
    graph = create_graph()
    print("Empty graph created.")
    
    graph = add_users_to_graph(graph, friends_and_follower_IDs)
    print("Graph updated with user nodes.")
    
    communities = girvan_newman(graph)
    print("Girvan Newman applied to graph. Communities received.")
    
    big_communities = []
    for eachCommunity in communities:
        #print(type(eachCommunity))
        if len(eachCommunity) >19:
            big_communities.append(eachCommunity)
    print("Now, we have a collection of big communities. The size of community has to be atleast 20.")        
    
    print("%d communities discovered with atleast 20 members"% len(big_communities))
        
    
    dict_clusters = {}
    clusterId = 0
    
    for eachCommunity in big_communities:
        
            #print(type(eachCommunity))
            dict_clusters[clusterId] = eachCommunity

            clusterId += 1

    
    print("Now, i have a dictionary. The key is a Numeric label (0,1,2...) and the value is a list of all members belonging to that cluster.")
    
 
    #Let's write this to a pkl file and later analyze, what is the sentiment in each community    
    
    output_file_dict_clusters = open('Cluster_Data/dict_clusters.pkl', 'wb')
    pickle.dump(dict_clusters,output_file_dict_clusters)
        
        
    
if __name__ == '__main__':
    main()