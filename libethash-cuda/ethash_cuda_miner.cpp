/*
  This file is part of libethash-cuda.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ethash_cuda_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015

Modified for Cuda by Ian Bloom
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <queue>
#include <random>
#include <vector>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include "ethash_cuda_miner.h"
