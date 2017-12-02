#!/usr/bin/env python3

import sys
from subprocess import Popen, PIPE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_FILE = '/tmp/output_points_0.txt'

exec_path = sys.argv[1]
distributed_write = (sys.argv[2] == '-d') if 2 < len(sys.argv) else False


def make_proc_args(exec_path, n_processes, n_points):
    args = [
        'mpiexec', '-n', str(n_processes), 'python3', exec_path,
        '-n', str(n_points), '-o', OUTPUT_FILE, '--chunk-size', '1024'
        ]
    if distributed_write:
        args.append('--distributed-write')
    return args


# heat up hdd

args = make_proc_args(exec_path, 4, 50000)
p = Popen(args, stderr=PIPE)
p.wait()


# testing

ns_proc = list(range(1, 9))
ns_points = list((10**m for m in range(8)))
num_tests = 5
test_data = []

for n_proc in ns_proc:
    test_data.append([])
    for n_points in ns_points:
        print(
            'Running tests with %i processes and %i points to generate'
            % (n_proc, n_points),
            file=sys.stderr
        )

        duration = 0
        for test_n in range(num_tests):
        
            # run
            args = make_proc_args(exec_path, n_proc, n_points)
            with Popen(args, stderr=PIPE) as proc:
                gen_output = proc.stderr.readline()
    
            # validate output
            with Popen(['wc', '-l', OUTPUT_FILE], stdout=PIPE) as wc:
                wc_output = wc.stdout.readline()
                assert n_points == int(wc_output.split()[0])
                
            duration += float(gen_output)

        duration /= num_tests
        test_data[-1].append(duration)


# post process and display data

test_data = np.array(test_data)
# normalized_test_data = test_data / test_data.min(axis=0)

df = pd.DataFrame(test_data, index=ns_proc, columns=ns_points)
# only for 1-4 processes
# norm_df = pd.DataFrame(normalized_test_data[:4, :], index=ns_proc[:4], columns=ns_points)

print()
print(df, end='\n\n')
print(df.T, end='\n\n')
# print(norm_df, end='\n\n')

df.T.plot(logx=True, logy=True)
plt.show()

# norm_df.plot.bar()
# plt.show()

