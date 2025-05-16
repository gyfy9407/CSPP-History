import os
os.environ["OMP_NUM_THREADS"] = "1"  # 修复内存泄漏警告
from pymoo.core.problem import Problem
import time
import csv
import os
from collections import defaultdict
import  numpy as np
from sklearn.ensemble import RandomForestRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random
import sys
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Integer
from pymoo.core.repair import Repair
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.ox import OrderCrossover
from scipy.stats.qmc import LatinHypercube
from pymoo.core.selection import Selection
from sklearn.cluster import KMeans,MiniBatchKMeans


class ClusteringTournamentSelection(Selection):
    def __init__(self, n_clusters=5, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters  # 每个Rank的聚类簇数

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        fronts = pop.get("rank")
        selected = []

        for rank in np.unique(fronts):
            current_front = pop[fronts == rank]
            X = current_front.get("X")

            # 如果当前前沿个体数不足，直接全部选择
            if len(current_front) <= self.n_clusters:
                selected.extend(current_front)
                continue

            # # K-means 聚类
            kmeans = KMeans(n_clusters=min(self.n_clusters, len(current_front)),
                            n_init=1,  # 减少初始化次数（加速）
                            algorithm="lloyd"  # 使用单线程算法
                            )  # 强制单线程,多线程处理小数据集时存在内存泄漏问题
            # kmeans = MiniBatchKMeans(
            #     n_clusters=min(self.n_clusters, len(current_front)),
            #     batch_size=256  # 根据数据大小调整
            # )
            clusters = kmeans.fit_predict(X)


            # 从每个簇中选择代表个体
            for c in np.unique(clusters):
                cluster_members = current_front[clusters == c]
                center = kmeans.cluster_centers_[c]
                dist = np.linalg.norm(cluster_members.get("X") - center, axis=1)
                representative = cluster_members[np.argmin(dist)]
                selected.append(representative)

        # 如果选出的代表不足，补充随机选择
        while len(selected) < n_select * n_parents:
            remaining = n_select * n_parents - len(selected)
            selected.extend(np.random.choice(pop, min(remaining, len(pop)), replace=False))

        # 确保返回的数组形状是 (n_select, n_parents)
        return np.array(selected[:n_select * n_parents]).reshape(n_select, n_parents)

class DualOutput:
    def __init__(self, file_name):
        self.console = sys.stdout
        self.file = open(file_name, 'a')# 追加写入

    def write(self, message):
        self.console.write(message)  # 输出到终端
        self.file.write(message)  # 输出到文件

    def flush(self):
        self.console.flush()
        self.file.flush()



def normalize(pop_obj_values, ref_pareto_front=None, data=None, lower_bound=[0, 0, 0], upper_bound=[300, 200, 100]):
    # 如果提供了 data，使用 pop_obj_values 和 data 的并集来计算最小值和最大值
    # if data is not None:
    #     # 合并 pop_obj_values 和 data
    #     combined_data = np.vstack((pop_obj_values, data))
    #     min_vals = np.min(combined_data, axis=0)  # 合并后的数据最小值
    #     max_vals = np.max(combined_data, axis=0)  # 合并后的数据最大值
    #     # print('使用 pop_obj_values 和 data 的并集的上下限进行归一化')
    # else:
    #     # 否则使用默认的上下限
    #     min_vals = np.array(lower_bound)
    #     max_vals = np.array(upper_bound)
    #     # print('使用默认的上下限进行归一化')
        # 否则使用默认的上下限
    min_vals = np.array(lower_bound)
    max_vals = np.array(upper_bound)
        # print('使用默认的上下限进行归一化')
    print('归一化上限：', max_vals)
    print('归一化下限：', min_vals)

    # 在归一化前过滤掉 ref_pareto_front 中超过上下限的点
    if ref_pareto_front is not None:
        valid_mask = np.all((ref_pareto_front >= min_vals) & (ref_pareto_front <= max_vals), axis=1)
        ref_pareto_front = ref_pareto_front[valid_mask]

    # 计算差值，防止 max_vals - min_vals 为 0 的情况
    diff = max_vals - min_vals
    diff[diff == 0] = 1  # 将差值为 0 的维度设置为 1，避免除零错误

    # 归一化 pop_obj_values
    normalized_pop = (pop_obj_values - min_vals) / diff

    # 如果提供了 ref_pareto_front，也对其进行归一化
    if ref_pareto_front is not None:
        normalized_ref = (ref_pareto_front - min_vals) / diff
        return normalized_pop, normalized_ref
    else:
        return normalized_pop


def calculate_igd(pop_obj_values, ref_pareto_front,data):
    def is_dominated(point, pareto_front):
        # 检查 point 是否被 pareto_front 中的某个点支配
        for front_point in pareto_front:
            if np.all(front_point <= point) and np.any(front_point < point):
                return True
        return False
    pop_obj_values=np.array(pop_obj_values)
    pop_obj_values, ref_pareto_front = normalize(pop_obj_values, ref_pareto_front,data)
    print(pop_obj_values)
    print(ref_pareto_front)
    # 添加 pop_obj_values 中的非支配解到 ref_pareto_front
    for point in pop_obj_values:
        if not is_dominated(point, ref_pareto_front):
            ref_pareto_front = np.vstack([ref_pareto_front, point])
    distances = []
    for ref_point in ref_pareto_front:
        min_distance = np.min(np.linalg.norm(pop_obj_values - ref_point, axis=1))
        distances.append(min_distance)
    return np.mean(distances)


def calculate_hv(pop_obj_values, ref_point,data):
    pop_obj_values = np.array(pop_obj_values)
    pop_obj_values = normalize(pop_obj_values,data=data)
    # print('pop_objective_value',pop_obj_values)
    if not np.all(np.less_equal(pop_obj_values, ref_point)):
        raise ValueError("pop_obj_values 中的点必须在每一维度上都不大于参考点 ref_point")
    # 确保帕累托前沿按第一个目标按降序排序
    pop_obj_values = pop_obj_values[pop_obj_values[:, 0].argsort()[::-1]]
    # 初始化超体积
    hv = 0.0
    # 逐步计算超体积
    for i in range(len(pop_obj_values)):
        if i == 0:
            width = ref_point[0] - pop_obj_values[i, 0]
        else:   # 递归体积分解?
            width = pop_obj_values[i - 1, 0] - pop_obj_values[i, 0]

        height = ref_point[1] - pop_obj_values[i, 1]
        depth = ref_point[2] - pop_obj_values[i, 2]

        # 计算三维体积（宽度 * 高度 * 深度）
        hv += width * height * depth
    return hv


# 将箱子按堆场街区 (C3) 和目的港 (Cp) 分组， 并做打乱
def getNewContainersList(CList): # 输入箱列表，输出新列表
    grouped_containers = defaultdict(list)
    virtual_containers = []

    # 根据isReal属性进行分类
    for container in CList:
        if container.isReal:
            grouped_containers[(container.C3, container.Cp)].append(container)
        else:
            virtual_containers.append(container)

    # 随机打乱每个组内的顺序
    for group in grouped_containers.values():
        random.shuffle(group)

    # 将组的键值打乱顺序
    group_keys = list(grouped_containers.keys())
    random.shuffle(group_keys)

    # 按打乱后的键值顺序将所有组合并为一个列表
    all_containers = [container for key in group_keys for container in grouped_containers[key]]

    # 随机插入虚拟箱子
    for virtual_container in virtual_containers:
        insert_position = random.randint(0, len(all_containers))
        all_containers.insert(insert_position, virtual_container)

    return all_containers

class Container:
    def __init__(self, id,weight,C3,Cp):
        self.id = int(id)
        self.weight = int(weight)  # 重量
        self.Cp = int(Cp)  # 目的港
        self.C3 = int(C3)  # 街区 l
        # self.setFlag=0  # 是否被分配位置
        self.isReal=True    # 是否虚拟箱
        self.setBay=None    # 分配到的贝位
        self.setStack=None  # 分配到的列
        self.setTier=None   # 分配到的层


class Bay:
    def __init__(self,id,num_s,num_t,w0):
        self.id=id
        self.Slot_40_Num=num_t*num_s    # 40普箱箱位数
        self.Container_Index=[] # 分配到贝位的箱
        self.H1_Difference=0  # // 与预配箱的重量差（稳定性）
        self.H1_Diversity =0  # // 混装贝位指标 = 目的港数 +堆场贝位数
        self.H3_reload=0    # f3,倒箱数
        self.Destinations=[]        # 贝中箱目的港
        self.YardBlocks=[]    # 贝中箱对应堆场街区
        self.W0=w0  # 预配重量列表，长度=贝内列数
        self.Deck=False # 是否甲板以上，1为是

NUMK=200

now=int(time.time())
folderPath=f'./testdataF{NUMK}C'
sys.stdout = DualOutput(f'C{NUMK}output{now}.txt')
csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]
for i,file in enumerate(csv_files[2:3],start=2):
    data = np.loadtxt(f'./reference/{NUMK}C_{i}.txt')
    file_path = os.path.join(folderPath, file)
    print(file_path)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        K_set = []  # 全部待配载箱
        for id,row in enumerate(reader):
            if id==0:
                numb=int(row[0])    # 船上贝位数
                nums=int(row[1] ) # 船，每贝位列数
                numt=int(row[2])  # 船，层数
                numk = int(row[3])  # 待配载箱数
                numl = int(row[4])  # 堆场街区数量
                nump = int(row[5])  # 目的港数
            elif id==1:     # 预配列重，内层list表示一个贝位，元素为列重
                W0 = [[int(row[j * nums + k]) for k in range(nums)] for j in range(numb)]
            else:   # 2行开始，每行表示一个箱，[id, 重量，所在堆场街区，目的港]
                K_set.append(Container(int(row[0]),int(row[1]),int(row[2]),int(row[3])))

    # 读完文件，整理
    P=[p for p in range(1,nump+1)] # 全部目的港（不能为0
    L=[l for l in range(1,numl+1)] # 全部堆场街区
    NP=100 # 上层种群规模
    sub_np=50  # 下层种群规模
    print('sub_np',sub_np)
    num_slot=numb*nums*numt # 船上箱位数=贝位数*列数*层数
    D=num_slot    # 维数=箱数（原为 按目的港的分组数
    for temp_count in range(num_slot-numk):  # 虚拟箱补全维数
        K_set.append(Container(temp_count+numk,0,0,0))    #[id, 重量，所在堆场街区，目的港]
        K_set[-1].isReal=False
    UPBOUND=100
    LOWBOUND=-100
    F=0.5
    CR=0.5
    VB_list=[]  # 全部船上贝位
    for b_id in range(numb):
        VB_list.append(Bay(b_id,nums,numt,W0[b_id]))    # 给的w0是4列
        if b_id<numb/2: # 前一半的贝位在甲板以上，待改
            VB_list[-1].Deck=True

    class Solution:
        def __init__(self):
            self.H1=0    # 与预配箱的重量差
            self.H2=0      # 混装指标
            self.H3=0   # 倒箱
            for b in VB_list:
                b.Container_Index=[]    # 分配到贝位的箱列表
                b.Destinations=[]       # 贝中箱的目的港
                b.Slot_40_Num=nums*numt # 40普箱箱位数=列数*层数
            self.Vessel_Bays_=VB_list # 船贝位
            self.List_Containers_=K_set # 所有集装箱
            self.Blocks_=L  # 堆场街区
            self.X1=[]  # 整数列表


    def Decoding_H2(xs): # 输入个体前半（贝位分箱），输出f2(混装指标）
        sol=Solution()
        for j in range(D):
            sol.X1.append(xs[j])    # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_=[]
        for i in range(D):
            Containers_.append(List_Containers_[int(sol.X1[i])])
        # 初始化子集列表
        num_subsets=int(numb/2)   # 分3个大贝位，甲板上下两个贝位为一个大贝位
        subsets = [[] for _ in range(num_subsets)]
        # 将箱均匀分配到子集中
        for i, container in enumerate(Containers_):
            subsets[i % num_subsets].append(container)
        sorted_subsets = []
        for subset in subsets:
            # 找出所有 isReal=True 的箱子并按 Cp 降序排列
            real_containers = sorted([container for container in subset if container.isReal],
                                     key=lambda x: x.Cp, reverse=True)
            # 找出所有 isReal=False 的箱子
            fake_containers = [container for container in subset if not container.isReal]
            # 创建一个新的子集，将排序后的 isReal=True 的箱子放在前面，isReal=False 的箱子放在后面
            new_subset = real_containers + fake_containers
            sorted_subsets.append(new_subset)
        for b in range(int(numb/2)):
            temp_lenth=len(sorted_subsets[b])//2
            bun=int(b+numb/2)
            Vessel_Bays_[b].Container_Index=sorted_subsets[b][temp_lenth:]  # 甲板上取后半,Cp小
            Vessel_Bays_[bun].Container_Index =sorted_subsets[b][:temp_lenth]   # 甲板下取前半
            for c in Vessel_Bays_[b].Container_Index:
                c.setBay=b
            for c in Vessel_Bays_[bun].Container_Index:
                c.setBay = bun

        for id, b in enumerate(Vessel_Bays_):
            Vessel_Bays_[id].Destinations=[]    # 贝位内箱的全部目的港，清空
            Vessel_Bays_[id].YardBlocks=[]      # 贝位内箱的全部堆场街区，清空
            for p in P: # 目的港
                if len(list(filter(lambda x:x.Cp==p,Vessel_Bays_[id].Container_Index)))>0:  # 如果有
                    Vessel_Bays_[id].Destinations.append(p)
            for l in L: # 堆场街区
                if len(list(filter(lambda x: x.C3 == l, Vessel_Bays_[id].Container_Index)) )> 0:
                    Vessel_Bays_[id].YardBlocks.append(l)

        # 评价贝位分配指标
        for id,b in enumerate(Vessel_Bays_):
            # 统计船舶贝位内集装箱的目的港和堆场贝位数
            b.H1_Diversity=len(b.Destinations)+len(b.YardBlocks)

        # 个体适应值更新
        sol.H2=0
        for i,_ in enumerate(Vessel_Bays_):
            sol.H2+=Vessel_Bays_[i].H1_Diversity
        return sol.H2   # 混装指标


    def sub_Decoding_pymoo(x, b, RealClist):
        def repair_vector(x, max_value=int(nums - 1), max_count=numt):
            x = np.round(x).astype(int)
            counts = np.zeros(max_value + 1, dtype=int)
            for i in range(len(x)):
                value = x[i]
                if counts[value] >= max_count:
                    new_value = (value + 1) % (max_value + 1)
                    while counts[new_value] >= max_count:
                        new_value = (new_value + 1) % (max_value + 1)
                    x[i] = new_value
                counts[x[i]] += 1
            return x

        x = repair_vector(x)

        column_weights = {i: 0 for i in range(nums)}
        causing_reload_boxes = set()  # 使用集合避免重复
        for i in range(len(x)):
            column = x[i]
            weight = RealClist[i].weight
            RealClist[i].setStack = column
            column_weights[column] += weight

        H1_differences = 0
        H3_reload = 0

        for column in range(nums):
            difference = (b.W0[column] - column_weights[column]) ** 2
            H1_differences += difference

            if b.Deck:
                temp_list = [c for c in RealClist if c.setStack == column]
                temp_list = sorted(temp_list, key=lambda c: (c.weight, c.Cp), reverse=True)
                for i in range(numt):
                    try:
                        temp_list[i].setTier = i
                    except IndexError:
                        continue
                for i, c1 in enumerate(temp_list[:-1]):
                    for c2 in temp_list[i + 1:]:
                        if c2.Cp > c1.Cp:
                            H3_reload += 1
                            causing_reload_boxes.add(c2.id)  # 添加到集合
                            # if c2.id not in causing_reload_boxes:
                            #     print(f"Adding box {c2.id} causing reload in column {column}")
                            #     causing_reload_boxes.append(c2.id)  # 避免重复添加
            else:
                temp_list = [c for c in RealClist if c.setStack == column]
                temp_list = sorted(temp_list, key=lambda c: c.Cp, reverse=True)
                for i in range(numt):
                    try:
                        temp_list[i].setTier = i
                    except IndexError:
                        continue

        b.H1_Difference = H1_differences
        b.H3_reload = H3_reload

        # 返回适应度值和导致倒箱的箱号
        return [H1_differences, H3_reload], list(causing_reload_boxes)

    class SubProblem(Problem):
        def __init__(self, num_variables, bay, container_list):
            super().__init__(n_var=num_variables, n_obj=2, xl=0, xu=int(nums - 1))
            self.bay = bay
            self.container_list = container_list
            self.reload_box_info = {}  # 存储每个解的导致倒箱的箱号列表

        def _evaluate(self, X, out, *args, **kwargs):
            fitness_values = []
            for i, x in enumerate(X):
                fitness, causing_boxes = sub_Decoding_pymoo(x, self.bay, self.container_list)
                fitness_values.append(fitness)
                self.reload_box_info[i] = causing_boxes  # 保存每个解的倒箱箱号
            out["F"] = np.array(fitness_values)


    class SequenceSamplingLHS(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            n_var = problem.n_var

            # 使用拉丁超立方采样生成 n_samples 个 n_var 维的样本
            sampler = LatinHypercube(d=n_var)
            lhs_samples = sampler.random(n=n_samples)

            # 对每个样本进行处理，将其映射为 0 ~ n_var-1 的整数序列
            samples = np.zeros((n_samples, n_var), dtype=int)
            for i in range(n_samples):
                samples[i] = np.argsort(lhs_samples[i])

            return samples

    class SequenceRepairWithRound(Repair):

        def _do(self, problem, X, **kwargs):
            nvar = problem.n_var  # 获取个体的变量数，即排列中的元素个数
            # 对每个个体进行修复
            for i in range(X.shape[0]):
                individual = X[i]  # 获取当前个体

                # 先进行取整操作
                individual = np.round(individual).astype(int)

                # 确保个体中的值在 [0, nvar-1] 范围内
                individual = np.clip(individual, 0, nvar - 1)

                # 确保个体中的元素不重复
                unique_values = np.unique(individual)
                missing_values = set(range(nvar)) - set(unique_values)  # 计算缺失的值

                # 初始化一个集合来追踪已处理的重复值
                replaced_values = set()

                # 通过替换重复的元素来修复个体
                for j in range(len(individual)):
                    if individual[j] in replaced_values:
                        continue  # 如果该值已经被替换过，则跳过

                    # 找出重复的元素
                    repeated_values = [val for val in individual if individual.tolist().count(val) > 1]

                    if repeated_values.count(individual[j]) > 1:
                        # 如果当前值在个体中是重复的，且未处理过，进行替换
                        if missing_values:  # 如果有缺失的元素
                            individual[j] = missing_values.pop()  # 替换该值为缺失的元素
                            replaced_values.add(individual[j])  # 记录替换过的值
                        else:
                            print(f"Warning: No more missing values to replace duplicates for individual {i}.")
                            break  # 如果没有更多的缺失值，跳出循环

                # 修复后的个体赋回
                X[i] = individual

            return X


    class PMX(Crossover):
        def __init__(self):
            # 定义 PMX 为两父母个体的交叉算子
            super().__init__(2, 2)

        def _do(self, problem, X, **kwargs):
            # 获取父母个体
            _, n_matings, n_var = X.shape
            # 存储子代
            children = np.full((2, n_matings, n_var), -1, dtype=int)

            for i in range(n_matings):
                parent1 = X[0, i]
                parent2 = X[1, i]

                # 随机选择交叉点
                point1, point2 = np.sort(np.random.choice(np.arange(n_var), 2, replace=False))

                # 子代1复制 parent1 的片段
                child1 = np.full(n_var, -1)
                child1[point1:point2 + 1] = parent1[point1:point2 + 1]

                # 子代2复制 parent2 的片段
                child2 = np.full(n_var, -1)
                child2[point1:point2 + 1] = parent2[point1:point2 + 1]

                # 构造映射关系
                mapping1 = {parent1[j]: parent2[j] for j in range(point1, point2 + 1)}
                mapping2 = {parent2[j]: parent1[j] for j in range(point1, point2 + 1)}

                # 修复 child1 中未分配的基因
                for j in range(n_var):
                    if child1[j] == -1:
                        gene = parent2[j]
                        while gene in mapping1:
                            gene = mapping1[gene]
                        child1[j] = gene

                # 修复 child2 中未分配的基因
                for j in range(n_var):
                    if child2[j] == -1:
                        gene = parent1[j]
                        while gene in mapping2:
                            gene = mapping2[gene]
                        child2[j] = gene

                # 保存子代
                children[0, i, :] = child1
                children[1, i, :] = child2

            return children

    # class AdaptiveCrossover(Crossover):
    #     def __init__(self, ox_crossover, pmx_crossover):
    #         super().__init__(2, 2)  # 2 parents, 2 offspring
    #         self.ox = ox_crossover
    #         self.pmx = pmx_crossover
    #         self.ox_prob = 0.7
    #         self.pmx_prob = 0.3
    #         self.ox_improvements = []
    #         self.pmx_improvements = []
    #         self.prev_fitness = None
    #
    #     def _do(self, problem, X, **kwargs):
    #         """
    #         执行交叉操作
    #
    #         Parameters:
    #         -----------
    #         problem : Problem
    #             优化问题实例
    #         X : ndarray
    #             父代个体，shape为(n_parents, n_matings, n_var)
    #
    #         Returns:
    #         --------
    #         ndarray
    #             子代个体，shape为(n_offsprings, n_matings, n_var)
    #         """
    #         # 获取当前种群的适应度值和更新概率
    #         current_fitness = kwargs.get("algorithm").pop.get("F")
    #         if self.prev_fitness is not None:
    #             improvements = current_fitness - self.prev_fitness
    #             mask_ox = kwargs.get("algorithm").pop.get("crossover_type") == "OX"
    #             mask_pmx = kwargs.get("algorithm").pop.get("crossover_type") == "PMX"
    #
    #             if np.any(mask_ox):
    #                 ox_improvement = np.mean(improvements[mask_ox], axis=0)
    #                 self.ox_improvements.append(ox_improvement)
    #             if np.any(mask_pmx):
    #                 pmx_improvement = np.mean(improvements[mask_pmx], axis=0)
    #                 self.pmx_improvements.append(pmx_improvement)
    #
    #             if len(self.ox_improvements) > 0 and len(self.pmx_improvements) > 0:
    #                 avg_ox_improvement = np.mean(self.ox_improvements[-1])
    #                 avg_pmx_improvement = np.mean(self.pmx_improvements[-1])
    #                 if avg_pmx_improvement > avg_ox_improvement:
    #                     self.pmx_prob = 0.7
    #                     self.ox_prob = 0.3
    #                 else:
    #                     self.pmx_prob = 0.3
    #                     self.ox_prob = 0.7
    #
    #         self.prev_fitness = current_fitness.copy() if current_fitness is not None else None
    #
    #         # 执行交叉操作
    #         n_matings = X.shape[1]
    #
    #         # 创建输出数组 - 确保形状符合pymoo的要求
    #         # shape: (n_offsprings, n_matings, n_var)
    #         Q = np.empty([self.n_offsprings, n_matings, problem.n_var], dtype=np.int64)
    #
    #         for i in range(n_matings):
    #             # 获取当前配对的父代
    #             current_parents = X[:, i:i + 1, :]  # shape: (2, 1, n_var)
    #
    #             # 根据概率选择交叉算子
    #             if np.random.random() < self.ox_prob:
    #                 # 使用OX交叉
    #                 off = self.ox._do(problem, current_parents)  # shape: (1, 2, n_var)
    #                 Q[:, i, :] = off[0]  # 将结果存入正确位置
    #                 if "algorithm" in kwargs and "pop" in kwargs["algorithm"].__dict__:
    #                     kwargs["algorithm"].pop[i].set("crossover_type", "OX")
    #             else:
    #                 # 使用PMX交叉
    #                 off = self.pmx._do(problem, current_parents)  # shape: (1, 2, n_var)
    #                 Q[:, i, :] = off[0]  # 将结果存入正确位置
    #                 if "algorithm" in kwargs and "pop" in kwargs["algorithm"].__dict__:
    #                     kwargs["algorithm"].pop[i].set("crossover_type", "PMX")
    #
    #         return Q
    class AdaptiveCrossover(Crossover):
        def __init__(self, ox_crossover, pmx_crossover):
            super().__init__(2, 2)  # 2 parents, 2 offspring
            self.ox = ox_crossover
            self.pmx = pmx_crossover
            self.ox_prob = 0.7
            self.pmx_prob = 0.3
            self.ox_improvements = []
            self.pmx_improvements = []
            self.prev_fitness = None
            self.last_update_gen = -1  # 初始化为无效值

        def _do(self, problem, X, **kwargs):
            """
            执行交叉操作

            Parameters:
            -----------
            problem : Problem
                优化问题实例
            X : ndarray
                父代个体，shape为(n_parents, n_matings, n_var)

            Returns:
            --------
            ndarray
                子代个体，shape为(n_offsprings, n_matings, n_var)
            """
            # 获取当前种群的适应度值和更新概率
            # current_fitness = kwargs.get("algorithm").pop.get("F")
            # 获取当前代数
            current_gen = kwargs.get("algorithm").n_gen

            # 确保每代只更新一次概率
            if current_gen != self.last_update_gen:
                self.last_update_gen = current_gen  # 更新记录
                current_fitness = kwargs.get("algorithm").pop.get("F")

                if self.prev_fitness is not None:
                    improvements = current_fitness - self.prev_fitness
                    mask_ox = kwargs.get("algorithm").pop.get("crossover_type") == "OX"
                    mask_pmx = kwargs.get("algorithm").pop.get("crossover_type") == "PMX"

                    if np.any(mask_ox):
                        ox_improvement = np.mean(improvements[mask_ox], axis=0)
                        self.ox_improvements.append(ox_improvement)
                    if np.any(mask_pmx):
                        pmx_improvement = np.mean(improvements[mask_pmx], axis=0)
                        self.pmx_improvements.append(pmx_improvement)

                    if len(self.ox_improvements) > 0 and len(self.pmx_improvements) > 0:
                        avg_ox_improvement = np.mean(self.ox_improvements[-1])
                        avg_pmx_improvement = np.mean(self.pmx_improvements[-1])
                        # print('ox_delta_f:', avg_ox_improvement)
                        # print('pmx_delta_f:', avg_pmx_improvement)
                        if avg_pmx_improvement > avg_ox_improvement:
                            self.pmx_prob = 0.7
                            self.ox_prob = 0.3
                        else:
                            self.pmx_prob = 0.3
                            self.ox_prob = 0.7
                        # print('ox_prob:', self.ox_prob)
                        # print('pmx_prob:', self.pmx_prob)
                self.prev_fitness = current_fitness.copy() if current_fitness is not None else None

            # 执行交叉操作
            n_matings = X.shape[1]

            # 创建输出数组 - 确保形状符合pymoo的要求
            # shape: (n_offsprings, n_matings, n_var)
            Q = np.empty([self.n_offsprings, n_matings, problem.n_var], dtype=np.int64)

            for i in range(n_matings):
                # 获取当前配对的父代
                current_parents = X[:, i:i + 1, :]  # shape: (2, 1, n_var)

                # 根据概率选择交叉算子
                if np.random.random() < self.ox_prob:
                    # 使用OX交叉
                    off = self.ox._do(problem, current_parents)  # shape: (1, 2, n_var)
                    Q[:, i, :] = off[0]  # 将结果存入正确位置
                    if "algorithm" in kwargs and "pop" in kwargs["algorithm"].__dict__:
                        kwargs["algorithm"].pop[i].set("crossover_type", "OX")
                else:
                    # 使用PMX交叉
                    off = self.pmx._do(problem, current_parents)  # shape: (1, 2, n_var)
                    Q[:, i, :] = off[0]  # 将结果存入正确位置
                    if "algorithm" in kwargs and "pop" in kwargs["algorithm"].__dict__:
                        kwargs["algorithm"].pop[i].set("crossover_type", "PMX")

            return Q


    class ProxyModelCallback(Callback):
        def __init__(self, max_archive_size=4 * NP):
            super().__init__()
            # self.archive = []  # 存档，用于存储已评估的解及其适应度
            self.max_archive_size = max_archive_size

        def notify(self, algorithm):
            """
            在每一代结束时调用：
            1. 将当前代的解加入存档
            2. 检查新解是否已评估，如果没有进行真实评估
            3. 使用存档更新代理模型
            4. 对非支配解执行邻域搜索
            """
            # 获取当前种群的决策变量和目标值（适应度）
            X = algorithm.pop.get("X")  # 当前种群的决策变量
            F = algorithm.pop.get("F")  # 当前种群的目标值（适应度）
            problem = algorithm.problem
            problem.generation = algorithm.n_gen
            # 将当前代的解添加到存档，并检查是否已评估
            for i in range(len(X)):
                true_fitness, solution = Decoding_Heuristic2(X[i])
                problem.archive.append((X[i], solution))  # 保存到存档
                F[i] = true_fitness  # 将真实评估结果赋值给当前解的适应度
                if not any(np.allclose(X[i], entry[0]) for entry in problem.archive):
                    # 如果解未评估过，进行真实评估并加入存档
                    true_fitness, solution = Decoding_Heuristic2(X[i])
                    problem.archive.append((X[i], solution))  # 保存到存档
                    F[i] = true_fitness  # 将真实评估结果赋值给当前解的适应度
                else:
                    # 解已评估，从存档中读取真实适应度
                    F[i] = np.round(next([entry[1].H1, entry[1].H2, entry[1].H3] for entry in problem.archive if
                                         np.allclose(X[i], entry[0], atol=1e-8))).astype(int)

            # 更新存档：确保存档大小不超过最大容量
            if len(problem.archive) > self.max_archive_size:
                problem.archive = problem.archive[-self.max_archive_size:]

            # 使用存档中的数据来更新代理模型
            # self.update_proxy_model(problem.archive)

            # 获取非支配解的索引
            nondominated_idx = self.get_nondominated_solutions(algorithm)

            for idx in nondominated_idx:
                solution = X[idx]
                for entry in problem.archive:
                    if np.array_equal(entry[0], solution):
                        # 获取倒箱箱号
                        causing_boxes = entry[1].causing_reload_boxes

                        # 执行第一种邻域搜索方案
                        X1_new = self.neighborhood_search1(solution)
                        fitness1, sol1 = Decoding_Heuristic2(X1_new)

                        # 执行第二种邻域搜索方案
                        X2_new, modified = self.neighborhood_search2(
                            entry[1], causing_boxes, solution.tolist()
                        )
                        fitness2, sol2 = (
                            Decoding_Heuristic2(X2_new) if modified else (F[idx], entry[1])
                        )

                        # 比较原解、解1、解2的适应度值
                        # candidates = [(F[idx], entry[1]), (fitness1, sol1), (fitness2, sol2)]

                        # 比较规则：逐维度比较
                        def is_better(fitness_current, fitness_new):
                            # fitness_current 和 fitness_new 都是一个多维向量
                            all_less_equal = all(f_new <= f_curr for f_new, f_curr in zip(fitness_new, fitness_current))
                            at_least_one_strictly_less = any(
                                f_new < f_curr for f_new, f_curr in zip(fitness_new, fitness_current))
                            return all_less_equal and at_least_one_strictly_less

                        # 初始化最优解为原解
                        best_fitness, best_solution = F[idx], entry[1]

                        # 遍历候选解，找到最优解
                        for fitness, solution in [(fitness1, sol1), (fitness2, sol2)]:
                            if is_better(best_fitness, fitness):  # 如果候选解支配当前最优解
                                best_fitness, best_solution = fitness, solution
                        # 如果最优解不是原解，替换当前种群解和存档
                        if not np.array_equal(best_fitness, F[idx]):  # 如果找到更优解
                            # print(f"解 {idx} 被邻域搜索改进")
                            F[idx] = best_fitness
                            X[idx] = (
                                X1_new if np.array_equal(best_fitness, fitness1) else X2_new
                            )  # 更新种群解
                            # 更新存档
                            problem.archive.append((X[idx], best_solution))
            algorithm.pop.set("F", F)

        def neighborhood_search1(self, X):
            """
            对个体的决策变量进行邻域搜索
            1. 将决策变量 X 均分为 int(numb/2) 段
            2. 随机交换各段的顺序
            :param X: 当前个体的决策变量（可以是种群的一个解）
            :return: 经过邻域搜索后的新的决策变量
            """
            n_var = len(X)  # 决策变量的维度
            num_segments = int(numb / 2)  # 将决策变量分为 int(numb/2) 段
            # 计算每段的大小
            segment_size = n_var // num_segments
            # 将 X 分成若干段
            segments = [X[i:i + segment_size] for i in range(0, n_var, segment_size)]
            # 随机打乱段的顺序
            np.random.shuffle(segments)
            # 将打乱顺序后的段重新拼接成一个新的解
            new_X = np.concatenate(segments)
            return new_X
        def neighborhood_search2(self, solution, causing_reload_boxes, X):
            """
            对解进行邻域搜索：
            - 遍历所有导致倒箱的箱（待改箱）。
            - 在甲板下寻找符合条件的箱（同Cp但更重），如果找到则在编码X上交换。
            - 返回更新后的X和是否发生改动的标志。
            """
            Vessel_Bays_ = solution.Vessel_Bays_
            # List_Containers_ = solution.List_Containers_
            modified = False

            for b in range(int(numb / 2)):  # 遍历所有大贝位
                bun = int(b + numb / 2)  # 甲板下对应贝位索引

                for container in Vessel_Bays_[b].Container_Index:
                    if container.id not in causing_reload_boxes:
                        continue  # 跳过非待改箱

                    cp = container.Cp
                    weight = container.weight

                    # 在甲板下寻找同Cp但更重的箱
                    for lower_container in Vessel_Bays_[bun].Container_Index:
                        if lower_container.Cp == cp and lower_container.weight > weight:
                            # 交换编码
                            idx1 = X.index(container.id)
                            idx2 = X.index(lower_container.id)
                            X[idx1], X[idx2] = X[idx2], X[idx1]
                            modified = True
                            break  # 一个待改箱只交换一次

            return X, modified


        def get_nondominated_solutions(self, algorithm):
            """
            获取当前种群的非支配解的索引
            :param algorithm: 当前的算法对象
            :return: 非支配解的索引列表
            """
            # 获取当前种群的目标值（适应度）
            F = algorithm.pop.get("F")  # 当前种群的目标值（适应度）

            # 使用 NonDominatedSorting 对目标值进行非支配排序
            nondominated_sorting = NonDominatedSorting()
            fronts = nondominated_sorting.do(F)  # 获取所有前沿

            # 获取第一个非支配前沿的解的索引（即非支配解）
            nondominated_idx = fronts[0]  # 直接使用 do() 方法返回的前沿，获取第一层非支配解

            return nondominated_idx



    class SwapMutation(Mutation):
        def __init__(self, prob=0.2):
            super().__init__()
            self.prob = prob

        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                if np.random.random() < self.prob:
                    # 随机选择两个不同的维度进行交换
                    dim1, dim2 = np.random.choice(X.shape[1], 2, replace=False)
                    X[i, [dim1, dim2]] = X[i, [dim2, dim1]]
            return X



    # nsga2
    def Decoding_Heuristic2(xs):
        sol = Solution()
        for j in range(D):
            sol.X1.append(xs[j])  # xs应该是启发式分箱顺序的箱号

        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_ = [List_Containers_[int(sol.X1[i])] for i in range(D)]

        # 初始化子集列表
        num_subsets = int(numb / 2)  # 分几个个大贝位，甲板上下两个贝位为一个大贝位
        subsets = [[] for _ in range(num_subsets)]
        for i, container in enumerate(Containers_):
            subsets[i % num_subsets].append(container)
        sorted_subsets = []
        for subset in subsets:
            real_containers = sorted([c for c in subset if c.isReal], key=lambda x: x.Cp, reverse=True)
            fake_containers = [c for c in subset if not c.isReal]
            sorted_subsets.append(real_containers + fake_containers)

        for b in range(int(numb / 2)):
            temp_lenth = len(sorted_subsets[b]) // 2
            bun = int(b + numb / 2)
            Vessel_Bays_[b].Container_Index = sorted_subsets[b][temp_lenth:]  # 甲板上取后半,Cp小
            Vessel_Bays_[bun].Container_Index = sorted_subsets[b][:temp_lenth]  # 甲板下取前半
            for c in Vessel_Bays_[b].Container_Index:
                c.setBay = b
            for c in Vessel_Bays_[bun].Container_Index:
                c.setBay = bun

        for id, b in enumerate(Vessel_Bays_):
            Vessel_Bays_[id].Destinations = []  # 贝位内箱的全部目的港，清空
            Vessel_Bays_[id].YardBlocks = []  # 贝位内箱的全部堆场街区，清空
            for p in P:  # 目的港
                if len(list(filter(lambda x: x.Cp == p, Vessel_Bays_[id].Container_Index))) > 0:  # 如果有
                    Vessel_Bays_[id].Destinations.append(p)
            for l in L:  # 堆场街区
                if len(list(filter(lambda x: x.C3 == l, Vessel_Bays_[id].Container_Index))) > 0:
                    Vessel_Bays_[id].YardBlocks.append(l)

        for id, b in enumerate(Vessel_Bays_):
            b.H1_Diversity = len(b.Destinations) + len(b.YardBlocks)

        sol.H2 = sum(b.H1_Diversity for b in Vessel_Bays_)

        n=10
        sol.H1 = 0
        sol.H3 = 0
        sol.causing_reload_boxes = []  # 用于记录所有子问题中最优解的倒箱箱号

        for b in Vessel_Bays_:
            real_container_list = [c for c in b.Container_Index if c.isReal]
            if not real_container_list:
                continue

            sub_problem = SubProblem(len(real_container_list), b, real_container_list)

            # 执行NSGA2算法解决子问题
            algorithm = NSGA2(pop_size=sub_np)
            res = minimize(sub_problem, algorithm, ('n_gen', n), seed=random.randint(0, 1000), verbose=False)

            # 获取最优解索引和对应的倒箱箱号
            best_idx = np.argmin([abs(f[0]) + abs(f[1]) for f in res.F])
            best_solution = res.F[best_idx]
            sol.H1 += best_solution[0]
            sol.H3 += best_solution[1]
            # 合并最优解的倒箱箱号，确保唯一性
            sol.causing_reload_boxes += sub_problem.reload_box_info[best_idx]
        # sol.causing_reload_boxes = list(sol.causing_reload_boxes)  # 转回列表
        Fitnessvalue = [sol.H1, sol.H2, sol.H3]
        return Fitnessvalue, sol


    # 定义问题
    class upLevelProblem2(Problem):
        def __init__(self, n_var):
            xl = np.zeros(n_var)  # 下限是0，长度为n_var
            xu = np.full(n_var, n_var - 1)  # 上限是n_var-1，长度为n_var
            super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu, type_var=Integer)
            self.archive = []
            self.generation = 0


        def _evaluate(self, X, out, *args, **kwargs):
            """
            适应度评估函数，使用代理模型预测适应度
            :param X: 当前解的决策变量，可能是多个解的矩阵 (n_pop, n_var)
            :param out: 输出字典，存储评估的结果
            """
            X = np.array(X)
            n_pop = X.shape[0]  # 获取种群大小
            F = np.zeros((n_pop, 3))  # 初始化适应度数组
            solutions = []  # 用于存储每个解对应的 Solution 对象

            for i in range(n_pop):
                x = X[i]
                true_fitness, solution = Decoding_Heuristic2(x)
                F[i] = true_fitness
                self.archive.append((x, solution))  # 存储完整的解和 Solution 对象
                solutions.append(solution)  # 保存 Solution 对象
                # 获取当前代数
                # if self.generation <= 0:
                #     # 第一代时，使用真实评估
                #     true_fitness, solution = Decoding_Heuristic2(x)
                #     F[i] = true_fitness
                #     self.archive.append((x, solution))  # 存储完整的解和 Solution 对象
                #     solutions.append(solution)  # 保存 Solution 对象
                # else:
                #     # 后续代数，使用代理模型预测
                #     predicted_fitness = self.predict_with_surrogate(x)
                #     F[i] = predicted_fitness
                #     solutions.append(None)  # 后续代数不保存 Solution 对象

            # 将适应度返回给 pymoo 框架
            out["F"] = F

            # 额外保存 Solution 对象，方便上层访问
            out["solutions"] = solutions

        # def predict_with_surrogate(self, x):
        #     """
        #     使用代理模型进行适应度预测
        #     假设有多个目标，可以分别使用代理模型进行预测
        #     """
        #     # 使用代理模型预测 f1 和 f3
        #     predicted_f1 = self.surrogate_models[0].predict([x])[0]
        #     predicted_f3 = self.surrogate_models[1].predict([x])[0]
        #     # 计算 f2
        #     calculated_f2 = Decoding_H2(x)
        #     # 返回 [f1, f2, f3] 的数组
        #     return np.array([predicted_f1, calculated_f2, predicted_f3])


    nsga2_sumset=[]
    nsga2_aver_f1=[]
    nsga2_aver_f2=[]
    nsga2_aver_f3=[]
    nsga2_aver_len_pareto=[]
    test_num=3   # 求平均
    N = 10  # 算法迭代次数
    print(time.time())
    for test in range(test_num):
        DB=[]
        DB_maxlen=int(NP*4) # DB最大容量
        print('--------------------------------nsga2-start----------------------------------')
        # surrogate_models = [RandomForestRegressor(n_estimators=100) for _ in range(2)]
        # 定义 OX 和 PMX 交叉算子
        ox_crossover = OrderCrossover()
        pmx_crossover = PMX()
        # 创建自适应交叉策略
        adaptive_crossover = AdaptiveCrossover(ox_crossover, pmx_crossover)
        problem = upLevelProblem2(n_var=D)
        start_time=time.time()
        algorithm = NSGA2(
            pop_size=NP,
            sampling=SequenceSamplingLHS(),
            repair=SequenceRepairWithRound(),
            selection=ClusteringTournamentSelection(n_clusters=5),  # 使用自定义选择
            mutation=SwapMutation(prob=0.5),  # 自定义的swap变异
            crossover=adaptive_crossover
        )        # 自定义的 sampling, crossover, mutation
        # 定义代理模型
        # 创建回调对象
        callback = ProxyModelCallback( max_archive_size=4*NP)
        # 调用 pymoo 的优化函数，传入回调
        res = minimize(
            problem,
            algorithm,
            ('n_gen', N),
            seed=random.randint(0, 1000),
            verbose=True,
            callback=callback
        )

        print("最终得到的解:")
        nsga2_solutions = np.unique(res.F, axis=0)  # 剔除重复解
        fitness_sums = [f[0] + f[1] + f[2] for f in nsga2_solutions]
        min_fitness_sum = min(fitness_sums)
        min_index = fitness_sums.index(min_fitness_sum)
        # nsga2_aver_f1.append(nsga2_solutions[min_index][0])
        # nsga2_aver_f2.append(nsga2_solutions[min_index][1])
        # nsga2_aver_f3.append(nsga2_solutions[min_index][2])
        # nsga2_aver_len_pareto.append(len(nsga2_solutions))
        # for f in nsga2_solutions:
        #     nsga2_sumset.append(f)
        end_time = time.time()
        print('nsga2time:', end_time - start_time)
        print('--------------------------------nsga2-end----------------------------------')
        print('nsga2_solutions:', nsga2_solutions)
        print('minF', nsga2_solutions[min_index])
        reference_point = [1.1, 1.1, 1.1]
        print('参考点：', reference_point)


        def is_dominated(sol, solutions):
            # 保证前沿上解互不支配
            return np.any(np.all(solutions <= sol, axis=1) & np.any(solutions < sol, axis=1))


        print('pareto_front:', data)
        nsga2_hv_value = calculate_hv(nsga2_solutions, reference_point, data)
        nsga2_igd_value = calculate_igd(nsga2_solutions, data, data)
        print('nsga2hv:', nsga2_hv_value)
        print('nsga2igd:', nsga2_igd_value)
