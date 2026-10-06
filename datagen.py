import os
import numpy
import argparse
from tqdm import tqdm
import const
import util

def sample_stargraph(path_len, n_paths):
    total_nodes_n = 1 + (path_len - 1) * n_paths
    start = numpy.random.randint(0, total_nodes_n)
    paths = []
    for _ in range(n_paths):
        paths.append([])
    next_path_idx = 0
    # populate nodes other than start into paths
    for i in range(total_nodes_n):
        if i == start:
            continue
        paths[next_path_idx].append(i)
        next_path_idx = (next_path_idx + 1) % n_paths
    # to string
    start = str(start)
    paths = [[str(node) for node in path] for path in paths]

    # edges formulation
    edges = []
    for i in range(path_len - 1):
        for j in range(n_paths):
            if i == 0:
                edges.append( (start, paths[j][i]) )
            else:
                edges.append( (paths[j][i-1], paths[j][i]) )
    # shuffle ordering
    edges = [ (b,a) if numpy.random.random() < 0.5 else (a,b)  for (a,b) in edges]
    numpy.random.shuffle(edges)

    # pick a path as tgt
    full_tgt_path = [start]
    for nd in paths[numpy.random.randint(0, n_paths)]:
        full_tgt_path.append(nd)

    # construct representation
    edges_tkns = const.EDGE_SEP.join([ const.GENERIC_COMMA.join(edge) for edge in edges ])
    begin_end = const.GENERIC_COMMA.join( (start, full_tgt_path[-1]) )
    tgt_seq = const.GENERIC_COMMA.join( full_tgt_path )
    # print(edges, start, full_tgt_path)
    return f"{edges_tkns}{const.START_END_PREFIX}{begin_end}{const.TGT_SEQ_PREFIX}{tgt_seq}{const.EOS}"


parser = argparse.ArgumentParser(
    prog="Datagen",
    description="Data generator args",
)
parser.add_argument("--path_len", "--pl", required=True, help="Path length")
parser.add_argument("--num_paths", "--np", required=True, help="Number of paths")
parser.add_argument("--num_samples", "--ns", required=True, help="Number of samples")
args, _ = parser.parse_known_args()

if __name__ == '__main__':
    pl = int(args.path_len)
    np = int(args.num_paths)
    ns = int(args.num_samples)
    dataset = []
    with tqdm(total=ns, desc="Data Sampling") as pbar:
        for _ in range(ns):
            dataset.append(sample_stargraph(pl, np))
            pbar.update(1)
    # save
    file_path = f"{pl}x{np}_{ns}.txt"
    full_path = util.get_abs_filepath(util.PATH_DATA, file_path)
    print(f"Saving data to {full_path}")
    print(f"Sample data: \n{dataset[0]}")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as file:
        file.write('\n'.join(dataset))