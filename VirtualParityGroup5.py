"""
ストレージシステムの容量最適化プログラム
物理ドライブと仮想ドライブの最適なマッピングを計算し、データ移動を最小化する
"""

from ortools.linear_solver import pywraplp
import itertools
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import time
import logging
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """ストレージシステムの設定を管理するデータクラス"""
    virtual_drives: int  # 仮想ドライブ数
    physical_drives: int  # 物理ドライブ数
    parcels_per_drive: int  # ドライブあたりのパーセル数
    
    @property
    def parcels_per_vdrive(self) -> int:
        """仮想ドライブあたりのパーセル数を計算"""
        return int(self.physical_drives * self.parcels_per_drive / self.virtual_drives)

@dataclass
class OptimizationVariables:
    """最適化問題の変数を保持するデータクラス"""
    solver: pywraplp.Solver
    variables: Dict  # ドライブ/スロット/パーセルごとの配置変数
    fill_rates: Dict  # パーセルごとの充填率
    config: Dict  # 最適化の設定パラメータ

class OptimizationProblem:
    """ストレージシステムの最適化問題における制約条件と目的関数を定義するクラス"""
    
    def __init__(self, solver: pywraplp.Solver):
        self.solver = solver
        self.vars = None
        self.objective = None

    def setup_variables(self, physical_drives: int, slots_per_drive: int, 
                       parcels: List[Tuple]) -> Dict:
        """最適化問題の変数を初期化"""
        variables = {}
        for drive in range(physical_drives):
            variables[drive] = {}
            for slot in range(slots_per_drive):
                variables[drive][slot] = {}
                for parcel in parcels:
                    variables[drive][slot][parcel] = self.solver.IntVar(0, 1, 
                        f'drive{drive}_slot{slot}_parcel{parcel}')
        return variables

    def add_total_parcels_constraint(self, vars: OptimizationVariables):
        """制約1: 全物理ドライブに配置されるパーセルの総数の制約"""
        total_parcels = len(vars.config['parcels'])
        self.solver.Add(
            sum(vars.variables[drive][slot][parcel]
                for drive in vars.variables
                for slot in vars.variables[drive]
                for parcel in vars.variables[drive][slot]) == total_parcels
        )

    def add_unique_parcel_constraint(self, vars: OptimizationVariables):
        """制約2: 同一パーセルの重複配置禁止"""
        for parcel in vars.config['parcels']:
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel]
                    for drive in vars.variables
                    for slot in vars.variables[drive]) == 1
            )

    def add_same_group_per_drive_constraint(self, vars: OptimizationVariables):
        """制約3: 同一物理ドライブ内の同一グループ番号制約"""
        for drive in vars.variables:
            for group_num in range(vars.config['parcels_per_vdrive']):
                self.solver.Add(
                    sum(vars.variables[drive][slot][parcel]
                        for slot in vars.variables[drive]
                        for parcel in vars.variables[drive][slot]
                        if parcel[1] == group_num) <= 1
                )

    def add_vdrive_parcel_count_constraint(self, vars: OptimizationVariables):
        """制約4: 仮想ドライブあたりのパーセル数制約"""
        for vdrive in vars.config['virtual_drives']:
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel]
                    for drive in vars.variables
                    for slot in vars.variables[drive]
                    for parcel in vars.variables[drive][slot]
                    if parcel[0] == vdrive) == vars.config['parcels_per_vdrive']
            )

    def add_parcel_group_distribution_constraint(self, vars: OptimizationVariables):
        """制約5: パーセルグループの分散配置制約"""
        for group_num in range(vars.config['parcels_per_vdrive']):
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel]
                    for drive in vars.variables
                    for slot in vars.variables[drive]
                    for parcel in vars.variables[drive][slot]
                    if parcel[1] == group_num) == vars.config['virtual_drives']
            )

    def add_physical_drive_capacity_constraint(self, vars: OptimizationVariables):
        """制約6: 物理ドライブの容量制約"""
        for drive in vars.variables:
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel]
                    for slot in vars.variables[drive]
                    for parcel in vars.variables[drive][slot]) == vars.config['parcels_per_drive']
            )

    def add_slot_allocation_constraint(self, vars: OptimizationVariables):
        """制約7: スロット割り当て制約"""
        for slot in range(vars.config['parcels_per_drive']):
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel]
                    for drive in vars.variables
                    for parcel in vars.variables[drive][slot]) == vars.config['physical_drives']
            )

    def add_single_parcel_per_slot_constraint(self, vars: OptimizationVariables):
        """制約8: スロットあたり1パーセル制約"""
        for drive in vars.variables:
            for slot in vars.variables[drive]:
                self.solver.Add(
                    sum(vars.variables[drive][slot][parcel]
                        for parcel in vars.variables[drive][slot]) == 1
                )

    def add_balanced_load_constraint(self, vars: OptimizationVariables):
        """制約9: 負荷分散制約"""
        total_fill = sum(vars.fill_rates.get(parcel, 1.0) 
                        for parcel in vars.config['parcels'])
        avg_fill = total_fill / vars.config['physical_drives']
        
        for drive in vars.variables:
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel] * vars.fill_rates.get(parcel, 1.0)
                    for slot in vars.variables[drive]
                    for parcel in vars.variables[drive][slot]) <= avg_fill + 1
            )
            self.solver.Add(
                sum(vars.variables[drive][slot][parcel] * vars.fill_rates.get(parcel, 1.0)
                    for slot in vars.variables[drive]
                    for parcel in vars.variables[drive][slot]) >= avg_fill - 1
            )

    def set_movement_minimization_objective(self, vars: OptimizationVariables, 
                                          current_layout: List[List[Tuple]]):
        """目的関数: データ移動量の最小化"""
        self.objective = self.solver.Objective()
        for drive in vars.variables:
            for slot in vars.variables[drive]:
                for parcel in vars.variables[drive][slot]:
                    movement_cost = 0 if (drive < len(current_layout) and 
                                        slot < len(current_layout[drive]) and 
                                        current_layout[drive][slot] == parcel) \
                                   else vars.fill_rates.get(parcel, 1.0)
                    self.objective.SetCoefficient(
                        vars.variables[drive][slot][parcel],
                        movement_cost
                    )
        self.objective.SetMinimization()

    def solve(self, vars: OptimizationVariables, current_layout: List[List[Tuple]]) -> bool:
        """最適化問題を解く"""
        self.add_total_parcels_constraint(vars)
        self.add_unique_parcel_constraint(vars)
        self.add_same_group_per_drive_constraint(vars)
        self.add_vdrive_parcel_count_constraint(vars)
        self.add_parcel_group_distribution_constraint(vars)
        self.add_physical_drive_capacity_constraint(vars)
        self.add_slot_allocation_constraint(vars)
        self.add_single_parcel_per_slot_constraint(vars)
        self.add_balanced_load_constraint(vars)
        
        self.set_movement_minimization_objective(vars, current_layout)
        
        status = self.solver.Solve()
        return status == pywraplp.Solver.OPTIMAL

class DatabaseManager:
    """データベース操作を管理するクラス"""
    
    def __init__(self, db_path: str):
        """データベース接続を初期化"""
        self.conn = sqlite3.connect(db_path)
        self._initialize_tables()
    
    def _initialize_tables(self):
        """必要なテーブルを作成"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    initial_drives INTEGER,
                    added_drives INTEGER,
                    total_drives INTEGER,
                    movement_cost REAL,
                    execution_time REAL,
                    moved_parcels INTEGER
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS drive_layouts (
                    result_id INTEGER,
                    drive_id INTEGER,
                    slot_id INTEGER,
                    parcel_id TEXT,
                    FOREIGN KEY (result_id) REFERENCES optimization_results(id)
                )
            ''')
    
    def save_result(self, result: dict) -> int:
        """最適化結果を保存しIDを返す"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO optimization_results 
                (timestamp, initial_drives, added_drives, total_drives,
                 movement_cost, execution_time, moved_parcels)
                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)
            ''', (result['initial_drives'], result['added_drives'],
                  result['total_drives'], result['movement_cost'],
                  result['execution_time'], result['moved_parcels']))
            return cursor.lastrowid
    
    def save_layout(self, result_id: int, layout: List[List[Tuple]]):
        """ドライブレイアウトを保存"""
        with self.conn:
            for drive_id, drive in enumerate(layout):
                for slot_id, parcel in enumerate(drive):
                    self.conn.execute('''
                        INSERT INTO drive_layouts
                        (result_id, drive_id, slot_id, parcel_id)
                        VALUES (?, ?, ?, ?)
                    ''', (result_id, drive_id, slot_id, str(parcel)))
    
    def close(self):
        """データベース接続を閉じる"""
        self.conn.close()

class StorageOptimizer:
    """ストレージシステムの最適化を行うメインクラス"""
    
    def __init__(self, config: StorageConfig):
        """初期化"""
        self.config = config
        self.current_layout = self._initialize_layout()
        self.fill_rates = {}
        self.db = DatabaseManager('storage_optimization.db')
        
    def _initialize_layout(self) -> List[List[Tuple]]:
        """初期レイアウトを生成"""
        parcels = self._generate_parcel_ids()
        return [parcels[i:i + self.config.parcels_per_drive] 
                for i in range(0, len(parcels), self.config.parcels_per_drive)]
    
    def _generate_parcel_ids(self) -> List[Tuple[str, int]]:
        """パーセルIDを生成"""
        virtual_drives = [chr(ord('A') + i) for i in range(self.config.virtual_drives)]
        return list(itertools.product(virtual_drives, range(self.config.parcels_per_drive)))

    def optimize(self) -> Tuple[List[List[Tuple]], float]:
        """最適化を実行"""
        solver = pywraplp.Solver.CreateSolver('SCIP')
        problem = OptimizationProblem(solver)
        
        # 変数の初期化
        variables = problem.setup_variables(
            self.config.physical_drives,
            self.config.parcels_per_drive,
            self._generate_parcel_ids()
        )
        
        # 最適化変数の設定
        opt_vars = OptimizationVariables(
            solver=solver,
            variables=variables,
            fill_rates=self.fill_rates,
            config={
                'parcels': self._generate_parcel_ids(),
                'virtual_drives': [chr(ord('A') + i) for i in range(self.config.virtual_drives)],
                'parcels_per_vdrive': self.config.parcels_per_vdrive,
                'parcels_per_drive': self.config.parcels_per_drive,
                'physical_drives': self.config.physical_drives
            }
        )
        
        # 最適化実行
        if problem.solve(opt_vars, self.current_layout):
            new_layout = self._extract_solution(variables)
            movement_cost = solver.Objective().Value()
            return new_layout, movement_cost
        else:
            raise RuntimeError("最適解が見つかりませんでした")
    
    def _extract_solution(self, variables: Dict) -> List[List[Tuple]]:
        """最適化結果からレイアウトを抽出"""
        layout = []
        for drive in range(self.config.physical_drives):
            drive_layout = []
            for slot in range(self.config.parcels_per_drive):
                for parcel in self._generate_parcel_ids():
                    if variables[drive][slot][parcel].solution_value() > 0.5:
                        drive_layout.append(parcel)
                        break
            layout.append(drive_layout)
        return layout
