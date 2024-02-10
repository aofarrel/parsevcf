import os
import argparse
import numpy as np
import ete3
import logging
import tqdm as progressbar

parser = argparse.ArgumentParser()
parser.add_argument('tree', type=str, help='path to MAT')
parser.add_argument('-s', '--samples', required=False, type=str,help='comma separated list of samples')
#parser.add_argument('-stdout', action='store_true', help='print matrix to stdout instead of a file')
parser.add_argument('-v', '--verbose', action='store_true', help='enable debug logging')
parser.add_argument('-nc', '--nocluster', action='store_true', help='do not search for clusters')
parser.add_argument('-o', '--out', required=False, type=str, help='what to append to output file name')
parser.add_argument('-d', '--distance', required=False, type=int, help='max distance between samples to identify as clustered')

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
tree = args.tree
t = ete3.Tree(tree, format=1)
if args.samples:
    samps = args.samples.split(',')
else:
    samps = sorted([leaf.name for leaf in t])
if args.out:
    prefix = args.out
else:
    prefix = os.path.basename(tree)
if args.distance:
    cluster_snp_distance = args.distance
else:
    cluster_snp_distance = 50

def path_to_root(ete_tree, node):
    # Browse the tree from a specific leaf to the root
    node = ete_tree.search_nodes(name=node)[0]
    path = [node]
    while node:
        node = node.up
        path.append(node)
    logging.debug(f"path for {node}: {path}")
    return path


def dist_matrix(tree, samples):
    samp_ancs = {}
    samp_dist = {}
    neighbors = []
    
    #for each input sample, find path to root and branch lengths
    for s in progressbar.tqdm(samples, desc="Finding roots and branch lengths"):
        s_ancs = path_to_root(tree, s)
        samp_ancs[s] = s_ancs
    
    #create matrix for samples
    matrix = np.full((len(samples),len(samples)), -1)

    for i in progressbar.trange(len(samples), desc="Creating matrix"): # trange is a tqdm optimized version of range
        s = samples[i]

        for j in range(len(samples)):
            os = samples[j]
            #Future goal: add catch to prevent reiteration of already checked pairs 
            if os == s:
                # self-to-self
                matrix[i][j] = '0'
                
            if os != s:
                if matrix[i][j] == -1: # ie, we haven't calculated this one yet
                    #find lca, add up branch lengths
                    s_path = 0 
                    os_path = 0 
                    
                    for a in samp_ancs[s]:
                        
                        s_path += a.dist
                        if a in samp_ancs[os]:
                        
                            lca = a
                            s_path -= a.dist
                            #logging.debug(f"found a in samp_ancs[os], setting s_path")
                            break
                    
                    for a in samp_ancs[os]:
                        
                        os_path += a.dist
                        if a == lca:
                            #logging.debug(f'a == lca, setting os_path')
                            os_path -= a.dist
                            break
                    logging.debug(f"sample {s} vs other sample {os}:")
                    logging.debug(f"s_path {s_path}, os_path {os_path}")
                    total_distance = int(s_path + os_path)
                    matrix[i][j] = total_distance
                    matrix[j][i] = total_distance
                    if not args.nocluster:
                        if total_distance <= cluster_snp_distance:
                            logging.debug(f"{s} and {os} might be in a cluster ({total_distance})")
                            neighbors.append(tuple((s, os)))
    if not args.nocluster:
        true_clusters = []
        first_iter = True
        for pairs in neighbors:
            existing_cluster = False
            if first_iter:
                true_clusters.append(set([pairs[0], pairs[1]]))
            else:
                for sublist in true_clusters:
                    if pairs[0] in sublist:
                        sublist.add(pairs[1])
                        existing_cluster = True
                    if pairs[1] in sublist: # NOT ELSE IF
                        sublist.add(pairs[0])
                        existing_cluster = True
                if not existing_cluster:
                    true_clusters.append(set([pairs[0], pairs[1]]))
            first_iter = False
    if args.nocluster:
        true_clusters = None
        
    return samples, matrix, true_clusters

samps, mat, clusters = dist_matrix(t, samps)

'''
for i in range(len(mat)):
    for j in range(len(mat[i])):
        if mat[i][j] != mat[j][i]:
            print(i,j)
'''

if not args.nocluster:
    # generate an auspice-style TSV for annotation of clusters
    with open(f"{prefix}clusters.tsv", "a") as cluster_tsv:
        cluster_tsv.write('Sample\tCluster\n')
        for i in range(len(clusters)):
            for sample in clusters[i]:
                cluster_tsv.write(f"{sample}\tcluster{i}\n")
    
    # generate another TSV for subtree annotation
    with open(f"{prefix}cluster_groups.tsv", "a") as groupped_clusters:
        groupped_clusters.write('Cluster\tSamples')
        for i in range(len(clusters)):
            groupped_clusters.write(f"\ncluster{i}\t")
            samples = ",".join([sample for sample in clusters[i]])
            groupped_clusters.write(f"{samples}")
    
    # recurse to get distance matrix of each cluster
    for i in range(len(clusters)):
        samples = ",".join([sample for sample in clusters[i]])
        print(samples)
        os.system(f"python3 distancematrix_nwk.py -s{samples} -nc -o cluster{i} '{tree}'")

with open(f"{prefix}distance_matrix.tsv", "a") as outfile:
    outfile.write(f'sample\t'+'\t'.join(samps))
    outfile.write("\n")
    for i in range(len(samps)):
        #strng = np.array2string(mat[i], separator='\t')[1:-1]
        line = [ str(int(count)) for count in mat[i]]
        outfile.write(f'{samps[i]}\t' + '\t'.join(line) + '\n')
        
