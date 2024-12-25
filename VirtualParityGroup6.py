# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import itertools
import sqlite3
import numpy as np
from ortools.linear_solver import pywraplp
import cupy as cp
from optparse import OptionParser
import time
import sys

@dataclass
class ParcelConfig:
    """パーセル設定を保持するデータクラス"""
    num_virtual_pdevs: int
    num_physical_pdevs: int
    num_physical_parcels: int
    num_virtual_parcels: int
    
    @classmethod
    def from_options(cls, options):
        """コマンドライン引数から設定を生成"""
        return cls(
            num_virtual_pdevs=options.initial,
            num_physical_pdevs=options.initial,
            num_physical_parcels=options.initial,
            num_virtual_parcels=options.initial
        )

class StorageOptimizer(ABC):
    """ストレージ最適化の基底クラス"""
    
    @abstractmethod
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        """最適化を実行する抽象メソッド"""
        pass

class ObjectiveFunction(StorageOptimizer):
    """目的関数を定義するクラス"""
    
    def __init__(self, current_pool: List, filled_data: Dict):
        self.current_pool = current_pool
        self.filled_data = filled_data

    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        """パーセル移動量を最小化する目的関数を設定"""
        objective = []
        for ppdev, ppdevparcel, parcel in variables.keys():
            if (self.current_pool[ppdev][ppdevparcel] != parcel and 
                self.current_pool[ppdev][ppdevparcel] != (None, None)):
                objective.append(variables[(ppdev, ppdevparcel, parcel)] * 
                               self.filled_data.get(parcel, 1.0))
        solver.Minimize(sum(objective))

class Constraint(StorageOptimizer):
    """制約条件の基底クラス"""
    def __init__(self, config: ParcelConfig):
        self.config = config

class TotalParcelsConstraint(Constraint):
    """全パーセル数に関する制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        total = sum(variables.values())
        solver.Add(total == self.config.num_virtual_pdevs * self.config.num_virtual_parcels)

class UniqueParcelConstraint(Constraint):
    """パーセルの重複を防ぐ制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        parcels = set(parcel for _, _, parcel in variables.keys())
        for parcel in parcels:
            total = sum(variables[(ppdev, ppdevparcel, p)]
                       for ppdev, ppdevparcel, p in variables.keys()
                       if p == parcel)
            solver.Add(total == 1)

class SamePhysicalPdevConstraint(Constraint):
    """同一物理PDEV内の制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        for ppdev in range(self.config.num_physical_pdevs):
            for vpdevparcel in range(self.config.num_virtual_parcels):
                total = sum(variables[(ppdev, ppdevparcel, parcel)]
                           for ppdevparcel in range(self.config.num_physical_parcels)
                           for _, vparcel in [parcel for _, _, parcel in variables.keys()]
                           if vparcel == vpdevparcel)
                solver.Add(total <= 1)

class VirtualPdevParcelCountConstraint(Constraint):
    """仮想PDEV当たりのパーセル数制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        vpdevs = set(vdev for vdev, _ in [parcel for _, _, parcel in variables.keys()])
        for vpdev in vpdevs:
            total = sum(variables[(ppdev, ppdevparcel, parcel)]
                       for ppdev, ppdevparcel, parcel in variables.keys()
                       if parcel[0] == vpdev)
            solver.Add(total == self.config.num_virtual_parcels)

class ParcelGroupCountConstraint(Constraint):
    """パーセルグループ数の制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        for vpdevparcel in range(self.config.num_virtual_parcels):
            total = sum(variables[(ppdev, ppdevparcel, parcel)]
                       for ppdev, ppdevparcel, parcel in variables.keys()
                       if parcel[1] == vpdevparcel)
            solver.Add(total == self.config.num_virtual_pdevs)

class PhysicalPdevLimitConstraint(Constraint):
    """物理PDEV容量制限の制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        for ppdev in range(self.config.num_physical_pdevs):
            total = sum(variables[(ppdev, ppdevparcel, parcel)]
                       for ppdevparcel, parcel in [(pp, p) 
                       for p, pp, pa in variables.keys()
                       if p == ppdev])
            solver.Add(total == self.config.num_physical_parcels)

class PhysicalLineConstraint(Constraint):
    """物理PDEVライン制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        for ppdevparcel in range(self.config.num_physical_parcels):
            total = sum(variables[(ppdev, ppdevparcel, parcel)]
                       for ppdev, parcel in [(p, pa)
                       for p, pp, pa in variables.keys()
                       if pp == ppdevparcel])
            solver.Add(total == self.config.num_physical_pdevs)

class SingleDataConstraint(Constraint):
    """単一データ配置制約"""
    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        for ppdev in range(self.config.num_physical_pdevs):
            for ppdevparcel in range(self.config.num_physical_parcels):
                total = sum(variables[(ppdev, ppdevparcel, parcel)]
                           for parcel in [p for _, _, p in variables.keys()])
                solver.Add(total == 1)

class LoadBalancingConstraint(Constraint):
    """負荷分散制約"""
    def __init__(self, config: ParcelConfig, filled_data: Dict):
        super().__init__(config)
        self.filled_data = filled_data

    def optimize(self, solver: pywraplp.Solver, variables: Dict) -> None:
        # 平均負荷を計算
        total_load = sum(self.filled_data.values())
        avg_load = total_load / self.config.num_physical_pdevs
        
        # 各物理PDEVの負荷を平均±1以内に制限
        for ppdev in range(self.config.num_physical_pdevs):
            load = sum(variables[(ppdev, ppdevparcel, parcel)] *
                      self.filled_data.get(parcel, 1.0)
                      for ppdevparcel in range(self.config.num_physical_parcels)
                      for parcel in [p for _, _, p in variables.keys()])
            solver.Add(load <= avg_load + 1)
            solver.Add(load >= avg_load - 1)

class ConstraintGenerator:
    """制約条件を生成するクラス"""
    
    def __init__(self, config: ParcelConfig, filled_data: Dict):
        self.config = config
        self.filled_data = filled_data
        self._constraints = []
        self._setup_constraints()

    def _setup_constraints(self) -> None:
        """全ての制約条件をセットアップ"""
        self._constraints = [
            TotalParcelsConstraint(self.config),
            UniqueParcelConstraint(self.config),
            SamePhysicalPdevConstraint(self.config),
            VirtualPdevParcelCountConstraint(self.config),
            ParcelGroupCountConstraint(self.config),
            PhysicalPdevLimitConstraint(self.config),
            PhysicalLineConstraint(self.config),
            SingleDataConstraint(self.config),
            LoadBalancingConstraint(self.config, self.filled_data)
        ]

    def apply_all_constraints(self, solver: pywraplp.Solver, variables: Dict) -> None:
        """全ての制約条件を適用"""
        for constraint in self._constraints:
            constraint.optimize(solver, variables)

class DKCOptimizer:
    """DKCシステムの最適化を行うクラス"""
    
    def __init__(self, config: ParcelConfig, use_gpu: bool = False):
        self.config = config
        self.use_gpu = use_gpu
        self.array_module = cp if use_gpu else np
        self.pool = self._initialize_pool()
        self.prev_pool = []
        self.filled_data = {}
        
    def _initialize_pool(self) -> List:
        """プールの初期化"""
        virtual_group = self._generate_virtual_parity_group()
        return [list(k) for k in zip(*[iter(virtual_group)]*self.config.num_virtual_parcels)]

    def _generate_virtual_parity_group(self) -> List:
        """仮想パリティグループの生成"""
        vpdevs = [chr(ord('A')+j) for j in range(self.config.num_virtual_pdevs)]
        vpdev_parcels = list(range(self.config.num_virtual_parcels))
        return list(itertools.product(vpdevs, vpdev_parcels))

    def fill_data(self, fill_ratio: float = 1.0) -> None:
        """データ充填処理"""
        def step_function(x: int, fill: float) -> float:
            return min(x, float(fill) * self.config.num_virtual_parcels)
        
        self.filled_data = {
            parcel: step_function(parcel[1]+1, fill_ratio) - step_function(parcel[1], fill_ratio)
            for ppdev in self.pool 
            for parcel in ppdev
        }

    def get_moved_parcels(self) -> List:
        """移動したパーセルのリストを取得"""
        if not self.prev_pool:
            return []
        
        moved = []
        for prev_parcel, current_parcel in zip(
            itertools.chain.from_iterable(self.prev_pool),
            itertools.chain.from_iterable(self.pool)
        ):
            if prev_parcel != current_parcel and prev_parcel != (None, None):
                moved.append(current_parcel)
        return moved

    def get_copy_amount(self) -> float:
        """コピー量を計算"""
        return sum(self.filled_data.get(parcel, 1.0) for parcel in self.get_moved_parcels())

    def get_distribution(self) -> float:
        """負荷分布を計算"""
        loads = []
        for ppdev in self.pool:
            load = sum(self.filled_data.get(parcel, 1.0) for parcel in ppdev)
            loads.append(load)
        return max(loads) - min(loads)

    def optimize_layout(self) -> None:
        """レイアウト最適化の実行"""
        solver = pywraplp.Solver.CreateSolver('SCIP')
        variables = self._create_variables(solver)
        
        # 目的関数と制約条件の設定
        objective = ObjectiveFunction(self.pool, self.filled_data)
        constraints = ConstraintGenerator(self.config, self.filled_data)
        
        objective.optimize(solver, variables)
        constraints.apply_all_constraints(solver, variables)
        
        # 最適化実行と結果の反映
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            self._update_pool(variables)
        else:
            raise RuntimeError("最適解が見つかりませんでした")

    def _create_variables(self, solver: pywraplp.Solver) -> Dict:
        """最適化変数の生成"""
        variables = {}
        parcels = self._generate_virtual_parity_group()
        for ppdev in range(self.config.num_physical_pdevs):
            for ppdevparcel in range(self.config.num_physical_parcels):
                for parcel in parcels:
                    name = f'x_{ppdev}_{ppdevparcel}_{parcel[0]}_{parcel[1]}'
                    variables[(ppdev, ppdevparcel, parcel)] = solver.IntVar(0, 1, name)
        return variables

    def _update_pool(self, variables: Dict) -> None:
        """プール構成の更新"""
        self.prev_pool = self.pool.copy()
        new_pool = []
        for ppdev in range(self.config.num_physical_pdevs):
            ppdev_parcels = []
            for ppdevparcel in range(self.config.num_physical_parcels):
                for parcel in self._generate_virtual_parity_group():
                    if variables[(ppdev, ppdevparcel, parcel)].solution_value() == 1:
                        ppdev_parcels.append(parcel)
            new_pool.append(ppdev_parcels)
        self.pool = new_pool

    def add_physical_pdevs(self, added_pdevs: int) -> None:
        """物理PDEVの追加"""
        self.config.num_physical_pdevs += added_pdevs
        self.config.num_virtual_parcels = self._calculate_virtual_parcels()
        # 空の物理PDEVを追加
        self.pool.extend([[(None, None) for _ in range(self.config.num_physical_parcels)] 
                         for _ in range(added_pdevs)])
    
    def _calculate_virtual_parcels(self) -> int:
        """仮想PDEV当たりのパーセル数を計算"""
        return int(self.config.num_physical_pdevs * 
                  self.config.num_physical_parcels / 
                  self.config.num_virtual_pdevs)

class SQLiteStorage:
    """SQLite3を使用したデータ永続化クラス"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
