#!/usr/bin/env python3


# = = = Ипорт MPI = = =


from mpi4py import MPI

Comm = MPI.COMM_WORLD
SIZE = Comm.Get_size()
RANK = Comm.Get_rank()


# = = = Парсинг и раздача аргументов = = =


import os
import sys
import argparse


args = None

if RANK == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, required=True)
    parser.add_argument('-o', '--output', type=str, default='-')
    parser.add_argument('--distributed-write', default=False, action='store_true')
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--radius', type=float, default=1.)
    args = parser.parse_args(sys.argv[1:])

    if os.path.isdir(args.output):
        print('ERROR: Output file is directory. Abort.')
        sys.exit(1)
    if os.path.isfile(args.output):
        open(args.output, 'w').close()

    directory = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(directory):
        os.makedirs(directory)

    args = {
        'number': args.number,
        'output': args.output,
        'distributed-write': args.distributed_write,
        'chunk-size': args.chunk_size,
        'radius': args.radius,
    }

args = Comm.bcast(args, root=0)
if RANK > 0 and args is None:
    sys.exit(1)


# = = = Константы и функции = = =

import pickle
from itertools import takewhile, count
from functools import partial

import numpy as np
import pandas as pd


DATA_POINTS_FORMATTED_TAG = 123
WRITE_PERMISSION_TAG = 124

DISTRIBUTED_WRITE = args['distributed-write']
OUTPUT_FILENAME = args['output']

POINTS_NUMBER = args['number']
RADIUS = args['radius']

CHUNK_SIZE = args['chunk-size']


def generate_points(amount):
    """Генерирует необходимое число точек с равномерным распределеннием по кругу"""
    angles = np.random.random(amount) * 2*np.pi
    distances = np.random.random(amount)
    distances = np.sqrt(distances) * RADIUS
    x = np.cos(angles) * distances
    y = np.sin(angles) * distances
    result = np.vstack([x, y]).T
    return result


# Самая быстрая, насколько смог, реализация форматирования массива точек в строку

def _generate_points_template(number):
    return '%.8f %.8f\n' * number


#FORMAT_CHUNK_SIZE = CHUNK_SIZE if DISTRIBUTED_WRITE else CHUNK_SIZE*SIZE
FORMAT_CHUNK_SIZE = CHUNK_SIZE
_full_chunk_template = _generate_points_template(FORMAT_CHUNK_SIZE)


def _pformat(points):
    if points.shape[0] == FORMAT_CHUNK_SIZE:
        return _full_chunk_template % tuple(points.flat)
    else:
        return _generate_points_template(points.shape[0]) % tuple(points.flat)


def _write_points_to_file(points):
    """Записывает точки в файл"""
    if OUTPUT_FILENAME == '-':
        print('\n'.join(_pformat(points)), file=sys.stdout)
        return
    else:
        with open(OUTPUT_FILENAME, 'a+') as f:
            # points_formatted = '\n'.join(_pformat(points)) + '\n'
            points_formatted = _pformat(points)
            f.write(points_formatted)


def _write_to_file(string):
    if OUTPUT_FILENAME == '-':
        print(string, file=sys.stdout)
    else:
        with open(OUTPUT_FILENAME, 'a+') as f:
            f.write(string)


def _save_points_distributed(chunk_idx, points):
    """Функция для сохранения точек распределенной записью.
    Каждый процесс при получении разрешения на запись записывает свои порцию из CHUNK_SIZE точек в файл
    и отправляет разрешение следующему."""
    _points_formatted = _pformat(points)
    # Если получено разрешение на запись в файл от предыдущего процесса
    if SIZE > 1:
        if Comm.recv(source=((SIZE+RANK-1) % SIZE), tag=WRITE_PERMISSION_TAG):
            _write_to_file(_points_formatted)
            # Разрешение на запись следущему
            if SIZE > 1 and chunk_idx + CHUNK_SIZE < POINTS_NUMBER:
                req = Comm.isend(1, dest=(RANK+1) % SIZE, tag=WRITE_PERMISSION_TAG)
                req.Free()
    else:
        _write_to_file(_points_formatted)


def _save_points_main(chunk_idx, points, _req=[None]):
    """Функция сохранения точек с централизованной записью.
    Каждый процесс отправляет процессу с рангом равным нулю свою порцию из CHUNK_SIZE точек.
    Процесс с нулевым рангом считывает каждую порцию и после записывает SIZE*CHUNK_SIZE точек."""
    if RANK != 0:
        # _req[0] это будто бы статическая переменная
        # а код ниже это своеобразный фикс от переполнения буфера
        if _req[0] is not None:
            _req[0].Wait()
        points_formatted = _pformat(points)
        _req[0] = Comm.isend(points_formatted, dest=0, tag=DATA_POINTS_FORMATTED_TAG)
        return
    _points_formatted = _pformat(points)
    for src_rank in range(1, SIZE):
        if chunk_idx + CHUNK_SIZE*src_rank < POINTS_NUMBER:
            _recv_formatted_points = Comm.recv(source=src_rank, tag=DATA_POINTS_FORMATTED_TAG)
            #_points = np.vstack([_points, _recv_points])
            _points_formatted += _recv_formatted_points
    _write_to_file(_points_formatted)


# Выбираю используемую функцию для сохранения точек
# в зависимости от DISTRIBUTED_WRITE флага
if DISTRIBUTED_WRITE:
    save_points = _save_points_distributed
    if SIZE > 1 and RANK == SIZE-1:
        # Последний процесс посылает разрешение на запись первого блока данных
        req = Comm.isend(1, dest=0, tag=WRITE_PERMISSION_TAG)
        req.Free()
else:
    save_points = _save_points_main


# = = = = = = = = Начало самой программы = = = = = = = =

Comm.Barrier()
time_start = MPI.Wtime()

for chunk_idx in takewhile(
            lambda x: x < POINTS_NUMBER,
            count(RANK*CHUNK_SIZE, SIZE*CHUNK_SIZE),
        ):
    amount_to_generate = min(CHUNK_SIZE, POINTS_NUMBER - chunk_idx)
    points = generate_points(amount_to_generate)
    save_points(chunk_idx, points)

Comm.Barrier()
time_spend = MPI.Wtime() - time_start

if RANK == 0:
    print(time_spend, file=sys.stderr)

