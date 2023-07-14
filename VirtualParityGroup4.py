# -*- coding: utf-8 -*-
import pulp, itertools, math
from collections import Counter
from functools import reduce
import sys
sys.path.append("C:\Cbc-2.9-win32-msvc14\bin")

class DKC(object):
    def __init__(self, numVpdevs=2, numPpdevs=2, numPpdevParcels=2):
        #物理PDEVの数
        self.numPpdevs=numPpdevs
        #仮想PDEVの数
        self.numVpdevs=numVpdevs
        #物理PDEV当たりのパーセル数
        self.numPpdevParcels=numPpdevParcels
        #仮想PDEV当たりのパーセル数
        self.numVpdevParcels=self.calc_numVpdevParcels()
        
        #全物理PDEVに連続してパーセルを充填するように初期化
        #zip(*[iter(data)]*n)を使って分割するとタプルになってしまうので、リスト構造に戻す
        self.Pool = [list(k) for k in zip(*[iter(self.getVirtualParityGroup())]*self.numVpdevParcels)]
        self.prevPool = []
        self.fillData(fill=1.0)

    #若いパーセルから順にページを充填
    def funcStep(self,x,fill):
        return min(x,float(fill)*self.numVpdevParcels)

    #ページをfuncStepで指定された法則に従い充填する
    def fillData(self,fill=1.0):
        func = self.funcStep
        self.Filled={parcel:func(parcel[1]+1,fill)-func(parcel[1],fill)
            for ppdev in self.Pool for parcel in ppdev}

    #物理PDEV毎のデータ充填量について、分散を求める
    #充填量最大と最小の差を分散と定義
    def getDifference(self):
        s = []
        for ppdev in self.Pool:
            s.append(sum([self.Filled.get(x,1.) for x in ppdev]))
        return max(s)-min(s)

    #プール構成を取得
    def getPool(self):
        return self.Pool

    def setPool(self,Pool):
        parcels = list(reduce(lambda a,b:a+b, Pool))
        #仮想PDEVの数は全パーセルでユニークな仮想PDEV番号の総数
        self.numVpdevs=len(set(map(lambda x:x[0],parcels)))
        #物理PDEVの数
        self.numPpdevs=len(Pool)
        #仮想PDEV当たりのパーセル数は全パーセルでユニークな仮想PDEV内のパーセル番号の総数
        self.numVpdevParcels=len(set(map(lambda x:x[1],parcels)))
        #物理PDEV当たりのパーセル数
        self.numPpdevParcels=max(map(lambda x:len(x),Pool))
        self.Pool = copy.deepcopy(Pool)
        self.prevPool = []
        self.fillData(fill=1.0)


    #前回の構成から移動が発生したパーセルを取得
    def getMoved(self):
        return list(map(lambda x:x[0], filter(lambda x:x[0] != x[1] and x[0] != (None,None),
                           zip(reduce(lambda a,b:a+b, self.prevPool),reduce(lambda a,b:a+b,self.Pool)))))

    #コピー発生したデータ量(ページ量)を求める
    def getCopyAmount(self):
        return sum([float(self.Filled[x]) for x in self.getMoved()])
                
    #物理PDEVを追加して、再計算する
    def addPpdevs(self,addedPpdevs=1):
        self.numPpdevs += addedPpdevs
        self.numVpdevParcels=self.calc_numVpdevParcels()
        #プールに空の物理PDEVを追加
        self.Pool.extend([[(None,None) for i in range(self.numPpdevParcels)] for j in range(addedPpdevs)])

    #物理PDEVを削除し、最大仮想PDEVパーセル番号を持つ物理PDEVパーセルをスペア領域に変更(PDEV障害を想定)
    def delPpdevs(self,Ppdev=0):
        spare = self.getVpdevParcels()[-1]
        self.numPpdevs -= 1
        self.numVpdevParcels=self.calc_numVpdevParcels()
        Pool = [ list(map(lambda x:x if not x[1]==spare else (None,None),Ppdev)) for Ppdev in self.Pool ]
        del Pool[Ppdev]
        self.Pool = Pool

    #仮想PDEVのID一覧を求める
    def getVpdevs(self):
        return [chr(ord('A')+j) for j in range(self.numVpdevs)]

    #仮想PDEV中のパーセルID一覧を求める
    def getVpdevParcels(self):
        return [j for j in range(self.numVpdevParcels)]


    #全パーセルのIDを求める
    def getVirtualParityGroup(self):
        #仮想PDEVの名前を、A,B,C...とつける
        Vpdevs = self.getVpdevs()
        #仮想PDEVのパーセルグループ名を、1,2,3,...と付ける
        VpdevParcels = self.getVpdevParcels()
        #両者の名前を組み合わせる
        return list(itertools.product(Vpdevs,VpdevParcels))
        

    #障害発生してもデータロストしないようにパーセルを再配置する
    def reconstruct(self):
        parcels = self.getVirtualParityGroup()
        ppdevs = range(self.numPpdevs)
        ppdevparcels = range(self.numPpdevParcels)
        vpdevs = set(list(zip(*parcels))[0])
        vpdevparcels = range(self.numVpdevParcels)

        #0-1整数計画問題を解く
        solver = pulp.LpVariable.dicts('Pool', (ppdevs,ppdevparcels,parcels), lowBound=0, upBound=1, cat=pulp.LpInteger)

        #目的関数
        #再配置に伴い、移動するパーセル数を最小化する
        parcelgroup_model = pulp.LpProblem("Parcel Group Corrision Avoidance", pulp.LpMinimize)
        parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                  *self.Filled.get(parcel,1.0)
                                  *(0 if (self.Pool[ppdev][ppdevparcel] == parcel or self.Pool[ppdev][ppdevparcel] == (None,None)) else 1)
                                  for ppdev in ppdevs
                                  for ppdevparcel in ppdevparcels
                                  for parcel in parcels])

        #制約条件
        #[条件1]全物理PDEVに配置されるパーセルの総数は、仮想PDEV数と仮想PDEV数あたりのパーセル数の積
        #これは条件6-8により自明なので要らないかも
        parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                  for ppdev in ppdevs
                                  for ppdevparcel in ppdevparcels
                                  for parcel in parcels]) == len(parcels)
        
        #[条件2]同一のIDを持つパーセルは複数存在しない
        for parcel in parcels:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for ppdev in ppdevs
                                      for ppdevparcel in ppdevparcels]) == 1

        #[条件3]同一の物理PDEVには同一のパーセルグループ番号を持つパーセルを配置しない
        for ppdev in ppdevs:
            for vpdevparcel in range(self.numVpdevParcels):
                parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                          for ppdevparcel in ppdevparcels
                                          for parcel in filter(lambda x:x[1]==vpdevparcel, parcels)]) <= 1

        #[条件4]全物理PDEVに配置されるパーセルの中で、同一の仮想PDEV番号を持つパーセルの総数は仮想PDEVあたりのパーセル数に等しい
        for vpdev in vpdevs:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for ppdev in ppdevs
                                      for ppdevparcel in ppdevparcels
                                      for parcel in filter(lambda x:x[0]==vpdev, parcels)]) == self.numVpdevParcels

        #[条件5]全物理PDEVに配置されるパーセルの中で、同一のパーセルグループ番号を持つパーセルの総数は仮想PDEV数に等しい(データロスト防止)
        for vpdevparcel in vpdevparcels:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for ppdev in ppdevs
                                      for ppdevparcel in ppdevparcels
                                      for parcel in filter(lambda x:x[1]==vpdevparcel, parcels)]) == self.numVpdevs

        #[条件6]各物理PDEVには配置可能な上限数のパーセルを配置する
        for ppdev in ppdevs:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for ppdevparcel in ppdevparcels
                                      for parcel in parcels]) == self.numPpdevParcels

        #[条件7]各物理PDEVの横のライン(名前未定)には配置可能な上限数のパーセルを配置する
        for ppdevparcel in ppdevparcels:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for ppdev in ppdevs
                                      for parcel in parcels]) == self.numPpdevs

        #[条件8]同一の領域に、2個以上のデータを入れてはいけない
        for ppdev in ppdevs:
            for ppdevparcel in ppdevparcels:
                parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      for parcel in parcels]) == 1

        #[条件9]物理PDEVに充填するページ量は平均値±1以内としてできるだけ均等化する
        ave = sum([self.Filled.get(x,1.) for x in reduce(lambda a,b:a+b,self.Pool)])/len(self.Pool)
        for ppdev in ppdevs:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      *self.Filled.get(parcel,1.)
                                      for ppdevparcel in ppdevparcels
                                      for parcel in parcels]) <= ave+1.
        for ppdev in ppdevs:
            parcelgroup_model += sum([solver[ppdev][ppdevparcel][parcel]
                                      *self.Filled.get(parcel,1.)
                                      for ppdevparcel in ppdevparcels
                                      for parcel in parcels]) >= ave-1.


        #計算実行
        parcelgroup_model.solve()

        #プール構成を変更
        Pool=[]
        for ppdev in ppdevs:
            tmp = []
            for ppdevparcel in ppdevparcels:
                for parcel in parcels:
                    if solver[ppdev][ppdevparcel][parcel].value() == 1.0:
                        tmp.append(parcel)
            Pool.append(tmp)
        self.prevPool = self.Pool
        self.Pool = Pool

    #仮想PDEV当たりのパーセル数を計算
    def calc_numVpdevParcels(self):
        return int(self.numPpdevs*self.numPpdevParcels/self.numVpdevs)
    
if __name__ == '__main__':
    import copy, time, redis, pickle
    from optparse import OptionParser
    host = '192.168.10.6'
    port = 6379    

    p=OptionParser()
    p.add_option('-n','--drives',default='4',type='int',dest='initial',help='Number of initial drives')
    p.add_option('-a','--add',default='1',type='int',dest='step',help='Number of additional drives')
    p.add_option('-m','--max',default='8',type='int',dest='final',help='Number of maximum drives')
    p.add_option('-d','--db',default='15',type='int',dest='db',help='ID of Redis DB')
    p.add_option('-L','--load',action='store',type='int',nargs=3,dest='load_key',help='Key to load from Redis DB')
    p.add_option('-K','--keys',action='store_true',default=False,dest='listing', help='Showing all keys')
    op, args = p.parse_args()

    conn = redis.Redis(host=host, port=port, db=op.db)
    if op.listing:
        for key in conn.keys():
            print(key)
        exit()

    current = op.initial
    storage=DKC(numVpdevs=op.initial, numPpdevs=op.initial, numPpdevParcels=op.initial)
    if not op.load_key == None:
        data = conn.get(list(op.load_key))
        if not data == None:
            load = pickle.loads(data)
            storage.setPool(load['pool_config'])
            current = load['stored_drives']
            op.initial = storage.numVpdevs
        else:
            exit()

    for k in range(current, op.final, op.step):
            storage.fillData(float(k-1)/float(k))
            storage.addPpdevs(addedPpdevs=op.step)

            t0 = time.clock()
            storage.reconstruct()
            t = time.clock()-t0
            
            print("Calculation Time: %f"%t)
            print("%d PDEV(s) were initially stored"%op.initial)
            print("%d PDEV(s) were newly added"%op.step)
            print("%d PDEV(s) were totally stored"%(k+1))

            moved = len(storage.getMoved())
            print("%d Parcel(s) were moved"%moved)
            print("%f CopyAmount"%storage.getCopyAmount())
            print("%f Distribution"%storage.getDifference())

            conn.set([op.initial,op.step,k+1],pickle.dumps({
                "cpu_time":t,
                "initial_drives":op.initial,
                "additional_drives":op.step,
                "stored_drives":k+1,
                "moved_parcel":storage.getMoved(),
                "pool_config":storage.getPool()}))
            for ppdev in storage.getPool():
                print(ppdev)
            print()

