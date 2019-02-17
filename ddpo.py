# -*- coding: utf-8 -*-
# Developed by Tomohiro Kawaguchi
# 2019/2/17
from ortools.sat.python import cp_model

class BinPacking:
    #ここら↓辺を参考に
    #https://qiita.com/SaitoTsutomu/items/3a970e071768bbe96005
    #https://en.wikipedia.org/wiki/Bin_packing_problem
    def __init__(self):
        self._bins = []
        self._items = []

    def addBin(self,capacity=0,ability=0):
        self._bins.append({'capacity':capacity,'ability':ability})

    def addItem(self,size=1,load=1):
        self._items.append({'size':size,'load':load})

    def _variable(self,model):
        #変数定義
        x = {}
        y = {}

        #荷物iを箱jに入れるか
        for i,item in enumerate(self._items):
            for j,bin in enumerate(self._bins):
                x[i,j] = model.NewBoolVar('x[%d,%d]'%(i,j))

        #箱jを使うか
        for j,bin in enumerate(self._bins):
            y[j] = model.NewBoolVar('y[%d]'%j)

        return x,y

    def _subject(self,model,x,y):
        #基本的な制約条件
        #制約条件
        #荷物iは何れかの箱に入れる
        for i,item in enumerate(self._items):
            model.Add(1 == sum(x[i,j] for j,bin in enumerate(self._bins)))

        #箱jの容量を超えない
        for j,bin in enumerate(self._bins):
            model.Add(bin['capacity'] >= sum(item['size']*x[i,j] for i,item in enumerate(self._items)))


    def _object(self,model,x,y):
        #目標関数
        #使用されている箱の総数を最小化する
        #箱jが使用されていない状態はy[j]=0
        #x[0..n,j]=0ならば、y[j]=0
        #x[0..n,j]の内で0となるものは1個以下のみであることを利用し、y[j]=sum(x[i,j]),i=0..nとする
        for j,bin in enumerate(self._bins):
            #model.AddBoolOr(y[j],(x[i,j] for i,item in enumerate(self._items)))
            model.Add(y[j]==sum(x[i,j] for i,item in enumerate(self._items)))
        model.Minimize(sum(y[j] for j,bin in enumerate(self._bins)))

    def _external(self,model,x,y):
        #追加制約条件
        pass

    def calculation(self):
        #モデル定義
        model = cp_model.CpModel()
        #変数定義
        x,y = self._variable(model)

        #基本的な制約条件の設定
        self._subject(model,x,y)

        #追加制約条件の設定
        self._external(model,x,y)

        #目的関数の設定
        self._object(model,x,y)

        solver = cp_model.CpSolver()
        state = solver.Solve(model)

        assert state != cp_model.MODEL_INVALID
        assert state != cp_model.UNKNOWN
        if state != cp_model.FEASIBLE and state != cp_model.OPTIMAL:
            return None
        else:
            return (k for k,v in x.items() if solver.Value(v)==1)

#目標関数を設定しない
class BinPackingEz(BinPacking):
    def _object(self,model,x,y):
        pass
    def _external(self,model,x,y):
        pass

#Abilityに制約
class BinPackingRe(BinPacking):
    def _object(self,model,x,y):
        pass
    def _external(self,model,x,y):
        #追加制約条件
        #箱jの能力を超えない
        for j,bin in enumerate(self._bins):
            model.Add(bin['ability'] >= sum(item['load']*x[i,j] for i,item in enumerate(self._items)))

#DDPOのための特殊計算
class BinPackingEx(BinPacking):
    def _object(self,model,x,y):
        #無駄(能力余剰と負荷過剰の二乗和)を最小にする組み合わせを求める
        dz={}
        ez={}
        pz={}
        for j,bin in enumerate(self._bins):
            dz[j] = model.NewIntVar(-0xFFFFFFFF,0xFFFFFFFF,'dz[%d]'%j)
            ez[j] = model.NewIntVar(          0,0xFFFFFFFFFFFFFFF,'ez[%d]'%j)
            pz[j] = model.NewIntVar(          0,0xFFFFFFFFFFFFFFF,'pz[%d]'%j)

            #能力と負荷の差の二乗を求める
            #一度に計算するとエラーになるので、展開して計算
            model.Add(dz[j]==sum(item['load']*x[i,j] for i,item in enumerate(self._items)))
            model.AddProdEquality(ez[j],[dz[j],dz[j]])
            model.Add(pz[j]==bin['ability']**2 - 2*bin['ability']*dz[j] + ez[j])
        '''
            #model.Add(dz[j]==bin['ability'] - sum(item['load']*x[i,j] for i,item in enumerate(self._items)))
            #model.AddProdEquality(pz[j],[dz[j],dz[j]])
        '''

        model.Minimize(sum(pz[j] for j,bin in enumerate(self._bins)))

    def _external(self,model,x,y):
        pass

class DDPO:
    def __init__(self):
        self.nodes=[]

    #'capacity':ノードの容量
    #'ability':ノードの計算能力
    #'size':ノードに格納されるデータの量
    #'load':ノードへのアクセスに伴う計算負荷
    #'transfer':超過データを他ノードに置くことによる、他ノードに与える負荷
    def addNode(self,capacity=1,ability=1,size=0,load=0,transfer=0):
        self.nodes.append({'capacity':capacity,'ability':ability,'size':size,'load':load,'transfer':transfer})

    def classifyNodes(self):
        #容量が充足しているノード(Remote)と、不足しているノード(に分類)
        remote = filter(lambda x:x[1]['capacity'] > x[1]['size'],enumerate(self.nodes))
        local  = filter(lambda x:x[1]['capacity'] <=x[1]['size'],enumerate(self.nodes))
        return list(remote),list(local)

    def splitChunks(self,chunks):
        k,v = max(enumerate(chunks), key=lambda x:x[1][1]['size'])
        #k,v = max(enumerate(chunks), key=lambda x:x[1]['size'])
        #全アイテムのサイズ(最大サイズ)が1ならばそれ以上分割できない
        if v[1]['size'] == 1:
            return False

        w = chunks.pop(k)

        import math
        size = math.ceil(w[1]['size'] / 2)
        load = math.ceil(w[1]['load'] / 2)

        chunks.append((w[0],{'size':size,'load':load}))
        chunks.append((w[0],{'size':w[1]['size']-size,'load':w[1]['load']-load}))
        return True

    def calculation(self):
        remotes,locals = self.classifyNodes()

        bins=[]
        for j,remote in remotes:
            bins.append((j,
                {'capacity':remote['capacity']-remote['size'],
                'ability':remote['ability']-remote['load']}))
        items=[]
        for i,local in locals:
            items.append((i,
                {'size':local['size']-local['capacity'],
                'load':local['transfer']}))

        while True:
            bp = BinPackingEx()
            for j,bin in bins:
                bp.addBin(**bin)
            for i,item in items:
                bp.addItem(**item)
            result = bp.calculation()
            if result!=None:
                break
            else:
                if not self.splitChunks(items):
                    print('Aborted')
                    assert True
                else:
                    print('Splitted')

        for info in result:
            f_node = items[info[0]][0]
            t_node = bins[info[1]][0]
            chunk  = items[info[0]][1]['size']
            print('%d chunks of node %d is placed to node %d'%(chunk,f_node,t_node))

if __name__ == '__main__':
    x=DDPO()
    x.addNode(capacity=100,ability=100,size=140,load=50,transfer=50)
    x.addNode(capacity=100,ability=100,size=140,load=50,transfer=50)
    x.addNode(capacity=100,ability=100,size=140,load=50,transfer=50)
    x.addNode(capacity=100,ability=100,size=140,load=50,transfer=50)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.addNode(capacity=160,ability=100,size=140,load=70,transfer=0)
    x.calculation()
