print("VERSION 1.4.7")
script_path = '/scripts/distancematrix_nwk.py'

import os
import argparse
import logging
from datetime import date
import numpy as np
import ete3
import tqdm as progressbar

parser = argparse.ArgumentParser()
parser.add_argument('tree', type=str, help='path to MAT')
parser.add_argument('-s', '--samples', required=False, type=str,help='comma separated list of samples')
parser.add_argument('-v', '--verbose', action='store_true', help='enable info logging')
parser.add_argument('-vv', '--veryverbose', action='store_true', help='enable debug logging')
parser.add_argument('-nc', '--nocluster', action='store_true', help='do not search for clusters')
parser.add_argument('-o', '--out', required=False, type=str, help='what to append to output file name')
parser.add_argument('-d', '--distance', default=20, type=int, help='max distance between samples to identify as clustered')
parser.add_argument('-nl', '--nolonely', action='store_true', help='if true, do not make a "cluster" of unclustered samples')

args = parser.parse_args()
if args.veryverbose:
    logging.basicConfig(level=logging.DEBUG)
elif args.verbose:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
tree = args.tree
t = ete3.Tree(tree, format=1)
samps = args.samples.split(',') if args.samples else sorted([leaf.name for leaf in t])
prefix = args.out if args.out else os.path.basename(tree).replace('.nwk', '').replace('_nwk', '')

def path_to_root(ete_tree, node_name):
    # Browse the tree from a specific leaf to the root
    logging.debug(f"Getting path for {node_name} in {type(ete_tree)}")
    node = ete_tree.search_nodes(name=node_name)[0]
    logging.debug(f"Node as found in ete tree: {node}")
    path = [node]
    while node:
        node = node.up
        path.append(node)
    logging.debug(f"path for {node_name}: {path}")
    return path


def dist_matrix(tree_to_matrix, samples):
    samp_ancs = {}
    #samp_dist = {}
    neighbors = []
    unclustered = set()
    
    #for each input sample, find path to root and branch lengths
    for sample in progressbar.tqdm(samples, desc="Finding roots and branch lengths"):
        s_ancs = path_to_root(tree_to_matrix, sample)
        samp_ancs[sample] = s_ancs
    
    #create matrix for samples
    matrix = np.full((len(samples),len(samples)), -1)

    for i in progressbar.trange(len(samples), desc="Creating matrix"): # trange is a tqdm optimized version of range
        this_samp = samples[i]
        definitely_in_a_cluster = False
        logging.debug(f"Checking {this_samp}")

        for j in range(len(samples)):
            that_samp = samples[j]
            #Future goal: add catch to prevent reiteration of already checked pairs
            if that_samp == this_samp: # self-to-self
                matrix[i][j] = '0'
            elif matrix[i][j] == -1: # ie, we haven't calculated this one yet
                #find lca, add up branch lengths 
                this_path = 0
                that_path = 0
                
                for a in samp_ancs[this_samp]:
                    this_path += a.dist
                    if a in samp_ancs[that_samp]:
                        lca = a
                        this_path -= a.dist
                        #logging.debug(f"  found a in samp_ancs[that_samp], setting this_path")
                        break
                
                for a in samp_ancs[that_samp]:
                    that_path += a.dist
                    if a == lca:
                        #logging.debug(f'  a == lca, setting that_path')
                        that_path -= a.dist
                        break
                
                logging.debug(f"  sample {this_samp} vs other sample {that_samp}: this_path {this_path}, that_path {that_path}")
                total_distance = int(this_path + that_path)
                matrix[i][j] = total_distance
                matrix[j][i] = total_distance
                if not args.nocluster and total_distance <= args.distance:
                    logging.debug(f"  {this_samp} and {that_samp} seem to be in a cluster ({total_distance})")
                    neighbors.append(tuple((this_samp, that_samp)))
                    definitely_in_a_cluster = True
        
        # after iterating through all of j, if this sample is not in a cluster, make note of that
        if not args.nocluster and not definitely_in_a_cluster:
            logging.debug(f"  {this_samp} is either not in a cluster or clustered early")
            #logging.debug(matrix[i])
            second_smallest_distance = np.partition(matrix[i], 1)[1] # second smallest, because smallest is self-self at 0
            if second_smallest_distance <= args.distance:
                logging.debug(f"  Oops, {this_samp} was already clustered! (closest sample is {second_smallest_distance}) SNPs away")
            else:
                logging.debug(f"  {this_samp} appears to be truly unclustered (closest sample is {second_smallest_distance} SNPs away)")
                unclustered.add(this_samp)
    
    # finished iterating, let's see what our clusters look like
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
    logging.debug("Returning:\n\tsamples:\n%s\n\tmatrix:\n%s\n\ttrue_clusters:\n%s\n\tunclustered:\n%s" % (samples, matrix, true_clusters, unclustered))
    return samples, matrix, true_clusters, unclustered

samps, mat, clusters, lonely = dist_matrix(t, samps)
total_samples_processed = len(samps)
logging.info(f"Processed {total_samples_processed} samples")
logging.debug(f"Samples processed: {samps}") # check if alphabetized

#for i in range(len(mat)):
#    for j in range(len(mat[i])):
#        if mat[i][j] != mat[j][i]:
#            print(i,j)

# this could probably be made more efficient
if not args.nocluster:

    logging.info("Clustering...")

    # sample_cluster is the Nextstrain-style TSV used for annotation, eg:
    # sample12    cluster1
    # sample13    cluster1
    # sample14    cluster1
    sample_cluster = ['Sample\tCluster\n']
    sample_clusterUUID = ['Sample\tClusterUUID\n']

    # cluster_samples is for matutils extract to generate Nextstrain subtrees, eg:
    # cluster1    sample12,sample13,sample14
    cluster_samples = ['Cluster\tSamples\n']

    # summary information for humans to look at
    n_clusters = len(clusters) # immutable
    n_samples_in_clusters = 0  # mutable
    
    for n in range(n_clusters):
        # get basic information -- we can safely sort here as do not use the array directly
        samples_in_cluster = sorted(list(clusters[n]))
        assert len(samples_in_cluster) == len(set(samples_in_cluster))
        n_samples_in_clusters += len(samples_in_cluster) # samples in ANY cluster, not just this one
        samples_in_cluster_str = ",".join(samples_in_cluster)
        is_cdph = any(samp_name[:2].isdigit() for samp_name in samples_in_cluster)
        UUID = str(args.distance).zfill(3) + "-" + str(n).zfill(5) + "-" + str(date.today())
        if is_cdph:
            # TODO: once we have metadata, switch to "California-YYYY"
            cluster_name = f"California-{UUID}"
            logging.info(f"{cluster_name}: CDPH, {len(samples_in_cluster)} members")
        else:
            # TODO: once we have metadata, switch to "ISO-YYYY"
            cluster_name = f"Open-{UUID}"
            logging.info(f"{cluster_name}: open, {len(samples_in_cluster)} members")

        # build cluster_samples line for this cluster
        cluster_samples.append(f"{cluster_name}\t{samples_in_cluster_str}\n")

        # recurse to matrix each cluster
        logging.debug(f"-->python3 {script_path} -s{samples_in_cluster_str} -vv -nc -o {prefix}_{cluster_name} '{tree}'")
        os.system(f"python3 {script_path} -s{samples_in_cluster_str} -vv -nc -o {prefix}_{cluster_name} '{tree}'")
        
        # build sample_cluster lines for this cluster - this will be used for auspice annotation
        for s in samples_in_cluster:
            sample_cluster.append(f"{s}\t{cluster_name}\n")
            sample_clusterUUID.append(f"{s}\t{UUID}\n")
    
    # add in the unclustered samples (outside for loop to avoid writing multiple times)
    if not args.nolonely:
        lonely = sorted(list(lonely))
        for george in lonely: # W0621, https://en.wikipedia.org/wiki/Lonesome_George
            sample_cluster.append(f"{george}\tlonely\n") # do NOT add to sample_clusterUUID lest persistent cluster IDs script break
        unclustered_as_str = ','.join(lonely)
        cluster_samples.append(f"lonely\t{unclustered_as_str}\n")
        cluster_samples.append("\n") # to avoid skipping last line when read
        logging.debug(f"-->python3 {script_path} -s{unclustered_as_str} -vv -nc -o {prefix}_lonely '{tree}'")
        os.system(f"python3 {script_path} -s{unclustered_as_str} -vv -nc -o {prefix}_lonely '{tree}'")

    # auspice-style TSV for annotation of clusters
    with open(f"{prefix}_cluster_annotation.tsv", "a") as samples_for_annotation:
        samples_for_annotation.writelines(sample_cluster)

    # auspice-style TSV with cluster UUIDs instead of full names; used for persistent cluster IDs
    with open(f"{prefix}_cluster_UUIDs.tsv", "a") as samples_by_cluster_UUID:
        samples_by_cluster_UUID.writelines(sample_clusterUUID)
    
    # usher-style TSV for subtree extraction
    with open(f"{prefix}_cluster_extraction.tsv", "a") as clusters_for_subtrees:
        clusters_for_subtrees.writelines(cluster_samples)
    
    # generate little summary files for WDL to parse directly
    with open("n_clusters", "w") as n_cluster: n_cluster.write(str(n_clusters))
    with open("n_samples_in_clusters", "w") as n_cluded: n_cluded.write(str(n_samples_in_clusters))
    with open("total_samples_processed", "w") as n_processed: n_processed.write(str(total_samples_processed))
    logging.info("Writing final matrix...") # keep this in the "if not args.nc" block; we don't want recursions to print it

with open(f"{prefix}_dmtrx.tsv", "a") as outfile:
    outfile.write('sample\t'+'\t'.join(samps))
    outfile.write("\n")
    for i in range(len(samps)): # don't change to enumerate without changing i; with enumerate it's a tuple
        #strng = np.array2string(mat[i], separator='\t')[1:-1]
        line = [ str(int(count)) for count in mat[i]]
        outfile.write(f'{samps[i]}\t' + '\t'.join(line) + '\n')
