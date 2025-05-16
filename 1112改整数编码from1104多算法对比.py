import time
import random
import copy
import csv
import os
from collections import defaultdict
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler    #  # 或者使用 MinMaxScaler
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
# problem = get_problem("zdt1")
import time
import random
import copy
import csv
import os
from collections import defaultdict
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler    #  # 或者使用 MinMaxScaler
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.algorithms.moo.mopso import MOPSO
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
import random
import sys
from pymoo.core.variable import BoundedVariable
# from pymoo.core.crossover import PMX
# from pymoo.core.mutation import SwapMutation
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Integer
from pymoo.core.repair import Repair
from deap import base, creator, tools
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



# def normalize(pop_obj_values, ref_pareto_front=None, lower_bound=[0,0,0], upper_bound=[300,100,100]):
#     min_vals = np.array(lower_bound)
#     max_vals = np.array(upper_bound)
#     print('归一化上限：',upper_bound)
#     # 在归一化前对 ref_pareto_front 进行过滤，删除超过 upper_bound 和 lower_bound 的点
#     if ref_pareto_front is not None:
#         valid_mask = np.all((ref_pareto_front >= min_vals) & (ref_pareto_front <= max_vals), axis=1)
#         ref_pareto_front = ref_pareto_front[valid_mask]
#
#     # 计算差值，防止 max_vals - min_vals 为 0 的情况
#     diff = max_vals - min_vals
#     diff[diff == 0] = 1  # 将差值为 0 的维度设置为 1，避免除零错误
#
#     # 归一化 pop_obj_values
#     normalized_pop = (pop_obj_values - min_vals) / diff
#
#     # 如果提供了参考帕累托前沿，也对其进行归一化
#     if ref_pareto_front is not None:
#         # 使用相同的 diff 对 ref_pareto_front 进行归一化
#         normalized_ref = (ref_pareto_front - min_vals) / diff
#         return normalized_pop, normalized_ref
#     else:
#         return normalized_pop
def normalize(pop_obj_values, ref_pareto_front=None, data=None, lower_bound=[0, 0, 0], upper_bound=[300, 200, 100]):
    # 如果提供了 data，使用 data 的最小值和最大值作为归一化上下限
    # if data is not None:
    #     min_vals = np.min(data, axis=0)
    #     max_vals = np.max(data, axis=0)
    #     # print('使用 data 的上下限进行归一化')
    # else:
    #     # 否则使用默认的上下限
    #     min_vals = np.array(lower_bound)
    #     max_vals = np.array(upper_bound)
    #     # print('使用默认的上下限进行归一化')
    min_vals = np.array(lower_bound)
    max_vals = np.array(upper_bound)
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
# scaler = StandardScaler()
def update_DB(new_elements):  # 更新用于训练代理模型的外部存档，保持不超过容量
    # 批量添加新元素
    DB.extend(new_elements)
    # 如果列表长度超过最大长度，移除多余的旧元素
    if len(DB) > DB_maxlen:
        DB[:] = DB[-DB_maxlen:]  # 只保留最后的 max_length 个元素



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

NUMK=300
now=int(time.time())
folderPath=f'./testdataF{NUMK}C'
sys.stdout = DualOutput(f'C{NUMK}output{now}.txt')
csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]
for i,file in enumerate(csv_files):
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

    class sub_MODE:
        def __init__(self,d,bay,Clist,):
            self.D=d
            self.b=bay
            self.Clist=Clist
            self.PopInit=[] # 当前种群
            self.PopMuta=[] # 变异种群
            self.PopCross=[]    # 子代种群
            self.PopPareto=[]   # 帕累托解集
            self.mode()
        def mode(self):
            for i in range(sub_np):
                self.PopInit.append(sub_Individual(self.D))
                self.PopInit[-1].Fitness=sub_Decoding(self.PopInit[-1].X,self.b,self.Clist) # 解码，获得适应度
                self.PopMuta.append(copy.deepcopy(self.PopInit[i]))
                self.PopCross.append(copy.deepcopy(self.PopInit[i]))
                self.PopPareto.clear()
        def Mutation(self):
            for i in range(sub_np):
                ##DE/rand(P)/1变异策略
                # 产生随机数
                while True:
                    randA = random.randint(0, sub_np-1)
                    randB = random.randint(0, sub_np-1)
                    if i!=randA and i!=randB and randA!= randB:
                        break
                randC=random.randint(0,len(self.PopPareto)-1)
                for j in range(self.D):
                    # print(self.PopInit[randA].X[j] - self.PopInit[randB].X[j])
                    self.PopMuta[i].X[j] = self.PopPareto[randC].X[j] + F * (self.PopInit[randA].X[j] - \
                                                                             self.PopInit[randB].X[j])
                    if self.PopMuta[i].X[j] > nums-1 or self.PopMuta[i].X[j] < 0:
                        self.PopMuta[i].X[j] = random.randint(0, nums-1)
                # self.PopMuta[i].repair_vector() # 修复，感觉这里有坑

        def Cross(self):
            for i in range(sub_np):
                for j in range(self.D):
                    if random.random()<CR:
                        self.PopCross[i].X[j] = self.PopMuta[i].X[j]
                    else:
                        self.PopCross[i].X[j] = self.PopInit[i].X[j]
                self.PopCross[i].repair_vector()
        def Select(self):
            for i in range(sub_np):
                # DE个体的解码，将实数向量转换成整数序列X
                self.PopCross[i].X_ = copy.deepcopy(self.PopCross[i].X)
                self.PopCross[i].Fitness=sub_Decoding(self.PopCross[i].X,self.b,self.Clist)
                if self.PopInit[i].Fitness[0] >= self.PopCross[i].Fitness[0] and\
                        self.PopInit[i].Fitness[1] >= self.PopCross[i].Fitness[1]:
                    self.PopInit[i] = copy.deepcopy(self.PopCross[i])
        def Pareto(self):
            for i in range(sub_np):
                DM=True # 默认非劣解
                if len(self.PopPareto)==0:  # 如果非劣解集为空，则将当前解加入
                    self.PopPareto.append(copy.deepcopy(self.PopInit[i]))
                else:
                    for j in range(len(self.PopPareto)):    # 遍历非劣解集中的解，分别与当前解进行比较
                        try:
                            if self.PopInit[i].Fitness[0] <= self.PopPareto[j].Fitness[0] and\
                                    self.PopInit[i].Fitness[1] <= self.PopPareto[j].Fitness[1]:
                                del self.PopPareto[j]   # 删除非劣解集中的解j
                                if len(self.PopPareto)==0 :  # 删空结束遍历
                                    break
                            elif self.PopInit[i].Fitness[0] > self.PopPareto[j].Fitness[0] and\
                                    self.PopInit[i].Fitness[1] > self.PopPareto[j].Fitness[1]:
                                DM = False #// 则当前解不是非劣解，标记位置false，结束遍历
                                break
                        except:
                            continue
                    if DM:
                        self.PopPareto.append(copy.deepcopy(self.PopInit[i]))
            return len(self.PopPareto)


    class MODE:
        def __init__(self,d):
            self.D=d
            self.PopInit = []  # 当前种群
            self.PopMuta = []  # 变异种群
            self.PopCross = []  # 子代种群
            self.PopPareto = []  # 帕累托解集
            self.mode()
            self.rf_models = [RandomForestRegressor(n_estimators=100, random_state=42) for _ in range(2)]
            self.X_trains = [np.empty((0, self.D)) for _ in range(2)]
            self.y_trains = [np.empty((0, 1)) for _ in range(2)]

        def mode(self):
            # 预生成一半随机生成的个体
            half_NP = NP // 2
            for i in range(half_NP, NP):
                individual = Individual(self.D)
                individual.Xs = list(range(self.D))  # Xs 是箱号列表，初始化为从 0 到 D-1 的顺序
                random.shuffle(individual.Xs)  # 随机打乱箱号列表
                # individual.X = individual.Xs.copy()  # 直接用乱序的箱号作为 X
                individual.Fitness = Decoding_Heuristic(individual.Xs)
                individual.RealFit = True
                self.PopInit.append(individual)
            # 生成前半部分的个体并进行初始化
            for i in range(half_NP):
                individual = Individual(self.D)
                self.PopInit.append(individual)
            # 将前面生成的个体添加到其他列表中
            for i in range(NP):
                self.PopMuta.append(copy.deepcopy(self.PopInit[i]))
                self.PopCross.append(copy.deepcopy(self.PopInit[i]))
            self.PopPareto.clear()

        def Mutation(self):
            for i in range(NP):
                while True:
                    randA = random.randint(0, NP - 1)
                    randB = random.randint(0, NP - 1)
                    if i != randA and i != randB and randA != randB:
                        break
                randC = random.randint(0, len(self.PopPareto) - 1)
                # 进行变异，基于箱号顺序交换位置
                for j in range(self.D):
                    try:
                        # 随机选择两个位置，交换它们的箱号
                        if random.random() < F:
                            idx1, idx2 = random.sample(range(self.D), 2)  # 随机选择两个索引
                            self.PopMuta[i].Xs[idx1], self.PopMuta[i].Xs[idx2] = self.PopMuta[i].Xs[idx2], \
                            self.PopMuta[i].Xs[idx1]
                        # 还可以进行其他变异方式，例如替换一个箱号等
                    except:
                        print('Error during mutation')

        def Cross(self):
            for i in range(NP):
                # 执行交叉操作
                for j in range(self.D):
                    if random.random() < CR:
                        # 从变异个体中复制箱号到交叉个体
                        self.PopCross[i].Xs[j] = self.PopMuta[i].Xs[j]
                    else:
                        # 保留原本的箱号
                        self.PopCross[i].Xs[j] = self.PopInit[i].Xs[j]

                # 修复重复的箱号
                self.fix_duplicate_boxes(self.PopCross[i])

        def fix_duplicate_boxes(self, individual):
            """
            修复个体中重复的箱号，确保每个箱号唯一
            """
            used_boxes = set()  # 用于记录已经使用的箱号
            duplicate_indices = []  # 用于记录重复箱号的索引
            all_boxes = set(range(self.D))  # 所有可能的箱号集合

            # 找出重复的箱号和缺失的箱号
            for j in range(self.D):
                box = individual.Xs[j]
                if box in used_boxes:
                    duplicate_indices.append(j)  # 记录重复的箱号的索引
                else:
                    used_boxes.add(box)  # 记录当前箱号为已用
                    all_boxes.discard(box)  # 移除已使用的箱号

            missing_boxes = list(all_boxes)  # 缺失的箱号，转换为列表

            # 替换重复的箱号
            for index in duplicate_indices:
                if missing_boxes:  # 如果有缺失的箱号
                    new_box = missing_boxes.pop()  # 获取一个缺失的箱号
                    individual.Xs[index] = new_box  # 用缺失的箱号替换重复的箱号
                    used_boxes.add(new_box)  # 更新已用箱号集合

            # 这时所有箱号都应该是唯一的，不会再出现重复

        def Select(self):
            for i in range(NP):
                # 复制 Xs 和 Xs_ 到新的属性，以保留交叉和变异操作之前的状态
                self.PopCross[i].Xs_ = copy.deepcopy(self.PopCross[i].Xs)
                # self.PopCross[i].decoding()  # 对新个体进行解码
                if I >= 1:  # I=0时模型未训练
                    # 预测适应度，现在基于 Xs 来计算
                    predicted_fitness = np.array(
                        [self.rf_models[j].predict([self.PopCross[i].Xs])[0] for j in range(2)])
                    self.PopCross[i].Fitness[0] = predicted_fitness[0]
                    self.PopCross[i].Fitness[1] = Decoding_H2(self.PopCross[i].Xs_)  # 基于解码后的 Xs
                    self.PopCross[i].Fitness[2] = predicted_fitness[1]
                    self.PopCross[i].RealFit = False
                else:
                    # 适应度计算基于 Xs
                    self.PopCross[i].Fitness = Decoding_Heuristic(self.PopCross[i].Xs)
                    self.PopCross[i].RealFit = True

            # 将初始种群和交叉种群组合成一个种群
            combined_population = copy.deepcopy(self.PopInit) + copy.deepcopy(self.PopCross)
            # 对组合种群进行非支配排序，得到不同的非支配层（帕累托前沿）
            fronts = self.non_dominated_sort(combined_population)
            # 初始化下一代种群
            next_population = []
            # 遍历每一个非支配层
            for front in fronts:
                # 如果当前非支配层可以全部加入下一代种群
                if len(next_population) + len(front) <= NP:
                    # 将当前非支配层中的所有个体加入下一代种群
                    next_population.extend(front)
                else:
                    self.crowding_distance_assignment(front)
                    # 按拥挤距离从大到小对当前非支配层进行排序
                    front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
                    # 选择拥挤距离最大的前几个个体填满下一代种群
                    next_population.extend(front[:NP - len(next_population)])
                    # 下一代种群已满，跳出循环
                    break

            # 确保下一代种群的大小为NP
            self.PopInit = next_population[:NP]

            # 对进入下一代的个体进行真实评估
            for ind in self.PopInit:
                if not ind.RealFit:
                    ind.Fitness = Decoding_Heuristic(ind.Xs)
                    ind.RealFit = True

            # 更新代理模型
            self.update_proxy_model()

        def update_proxy_model(self):
            # 更新代理模型，分别处理第一维和第三维
            indices_to_use = [0, 2]  # 只使用第一维和第三维
            for i, idx in enumerate(indices_to_use):
                self.X_trains[i] = np.array([ind.Xs for ind in DB])  # 使用 Xs 作为特征输入
                self.y_trains[i] = np.array([ind.Fitness[idx] for ind in DB]).reshape(-1, 1)
                self.rf_models[i].fit(self.X_trains[i], self.y_trains[i].ravel())

        def dominates(self, p, q):
            # 初始状态，假设没有任何目标是更好的
            better_in_any = False
            # 遍历每个目标值，比较p和q的适应度值
            for i in range(len(p.Fitness)):
                # 如果p在该目标上优于q
                if p.Fitness[i] < q.Fitness[i]:
                    better_in_any = True
                # 如果p在该目标上劣于q，则p不支配q
                elif p.Fitness[i] > q.Fitness[i]:
                    return False
            # 如果p在某个目标上优于q，并且没有在其他目标上劣于q，则p支配q
            return better_in_any

        def non_dominated_sort(self, population):
            """
            非支配排序：将种群分成不同的支配层，每层内的个体互不支配
            """
            # fronts用于存储所有支配层的列表
            fronts = [[]]
            # 初始化每个个体的支配解集和支配计数
            for p in population:
                p.dominated_solutions = []  # 被p支配的解
                p.domination_count = 0  # 支配p的解数量
                # 比较p和种群中的每一个q
                for q in population:
                    if q is not p:
                        if self.dominates(p, q):    # 如果p支配q
                            p.dominated_solutions.append(q)
                        elif self.dominates(q, p):  # 如果q支配p
                            p.domination_count += 1
                # 如果p没有被任何解支配，将其放入第一个前沿（第一层）
                if p.domination_count == 0:
                    p.rank = 0
                    fronts[0].append(p)
            # 迭代创建下一个支配层
            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for p in fronts[i]:
                    for q in p.dominated_solutions:
                        q.domination_count -= 1
                        if q.domination_count == 0:
                            q.rank = i + 1
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            # 去掉最后一个空的支配层
            fronts = fronts[:-1]
            return fronts

        def crowding_distance_assignment(self, front):
            """
            拥挤距离分配：用于在同一支配层中区分个体的选择优先级
            """
            if len(front) == 0:
                return
            # 目标的数量
            num_objectives = len(front[0].Fitness)
            # 初始化每个个体的拥挤距离为0
            for individual in front:
                individual.crowding_distance = 0
            # 对于每个目标函数，计算个体的拥挤距离
            for m in range(num_objectives):
                # 根据目标m的适应度值对个体进行排序
                front.sort(key=lambda x: x.Fitness[m])
                # 将边界个体的拥挤距离设为无穷大，以确保它们被选中
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                # 计算当前目标的最大和最小适应度值
                min_fitness = front[0].Fitness[m]
                max_fitness = front[-1].Fitness[m]
                # 如果最大值和最小值相等，则跳过计算
                if max_fitness - min_fitness == 0:
                    continue
                # 计算内部个体的拥挤距离
                for i in range(1, len(front) - 1):
                    # 拥挤距离是通过邻居的目标差异归一化计算得出的
                    front[i].crowding_distance += (front[i + 1].Fitness[m] - front[i - 1].Fitness[m]) / (
                                max_fitness - min_fitness)

        def Pareto(self):
            for i in range(NP):
                DM = True  # 默认非劣解
                if len(self.PopPareto) == 0:  # 如果非劣解集为空，则将当前解加入
                    self.PopPareto.append(copy.deepcopy(self.PopInit[i]))
                else:
                    j = 0
                    while j < len(self.PopPareto):  # 使用while循环手动控制索引
                        if (self.PopInit[i].Fitness[0] <= self.PopPareto[j].Fitness[0] and
                                self.PopInit[i].Fitness[1] <= self.PopPareto[j].Fitness[1] and
                                self.PopInit[i].Fitness[2] <= self.PopPareto[j].Fitness[2]):
                            del self.PopPareto[j]  # 删除非劣解集中的解j
                            if len(self.PopPareto) == 0:  # 删空结束遍历
                                break
                        elif (self.PopInit[i].Fitness[0] >= self.PopPareto[j].Fitness[0] and
                              self.PopInit[i].Fitness[1] >= self.PopPareto[j].Fitness[1] and
                              self.PopInit[i].Fitness[2] >= self.PopPareto[j].Fitness[2]):
                            DM = False  # 当前解不是非劣解，标记为false，结束遍历
                            break
                        else:
                            j += 1  # 只有在没有删除时才增加索引
                    if DM:
                        self.PopPareto.append(copy.deepcopy(self.PopInit[i]))
            return len(self.PopPareto)


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

    def sub_Decoding(x,b,RealClist): # 箱位分箱解码,个体向量、预配列重、箱列表
        # w0四列，x匹配到箱重，作差求和
        column_weights = {i: 0 for i in range(nums)}  # 假设列号范围是0~3
        for i in range(len(x)):
            column = x[i]  # 取x向量中当前箱子的列号
            weight = RealClist[i].weight  # 取RealClist中对应箱子的重量
            RealClist[i].setStack=column    # 箱分配列号
            column_weights[column] += weight  # 将重量累加到对应的列号中
        H1_differences = 0
        H3_reload=0
        for column in range(nums):  # 列重差求和
            difference = (b.W0[column] - column_weights[column])**2
            H1_differences+=difference
            if b.Deck:  # 如果甲板以上，按重量降序排,重量相同时目的港降序
                temp_list=[c for c in RealClist if c.setStack==column]
                temp_list = sorted(temp_list, key=lambda c: (c.weight, c.Cp), reverse=True)
                for i in range(numt):
                    try:
                        temp_list[i].setTier=i
                    except:
                        continue
                for i, c1 in enumerate(temp_list[:-1]):
                    for c2 in temp_list[i + 1:]:
                        if c2.Cp>c1.Cp: # 因目的港的倒箱
                            H3_reload+=1
            else:
                temp_list=[c for c in RealClist if c.setStack==column]
                temp_list = sorted(temp_list, key=lambda c: c.Cp, reverse=True) # 甲板下，按目的港降序
                for i in range(numt):
                    try:
                        temp_list[i].setTier=i
                    except:
                        continue
        b.H1_Difference=H1_differences
        b.H3_reload=H3_reload
        return [H1_differences,H3_reload]

    def Decoding_Heuristic(xs): # 输入个体前半（贝位分箱），输出f2(混装指标）、贝位分到的箱
        sol=Solution()
        for j in range(D):
            sol.X1.append(xs[j])    # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_=[]
        for i in range(D):
            Containers_.append(List_Containers_[sol.X1[i]])
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
        # 箱分到箱位
        n=10    # 子问题迭代轮次
        sol.H1=0
        sol.H3=0
        # best_individual_list=[]
        for b in Vessel_Bays_:  # 对每个贝
            real_container_count = sum(1 for container in b.Container_Index if container.isReal)    # 分到贝位的真实箱数量，等于个体维度
            real_container_list = [container for container in b.Container_Index if container.isReal]    # 真实箱列表
            subMde=sub_MODE(real_container_count,b,real_container_list)
            subMde.Pareto()
            for I in range(n):
                subMde.Mutation()
                subMde.Cross()
                subMde.Select()
                subMde.Pareto()
            # 找到帕累托前沿中距(0,0)曼哈顿距离最小的个体
            min_manhattan_distance = float('inf')
            best_individual = None
            for individual in subMde.PopPareto:
                manhattan_distance = abs(individual.Fitness[0]) + abs(individual.Fitness[1])
                if manhattan_distance < min_manhattan_distance:
                    min_manhattan_distance = manhattan_distance
                    best_individual = individual
            sol.H1 += best_individual.Fitness[0]
            sol.H3 += best_individual.Fitness[1]
        Fitnessvalue=[sol.H1,sol.H2,sol.H3]
        return Fitnessvalue

    def outcome_Decoding_Heuristic(xs): # 只用于最后输出，和Decoding_Heuristic基本相同，多了一个返回值
        def outcome_sub_Decoding(x,b,RealClist):
            for i in range(len(x)):
                column = x[i]  # 取x向量中当前箱子的列号
                RealClist[i].setStack = column  # 箱分配列号
            for column in range(nums):  # 列重差求和
                if b.Deck:  # 如果甲板以上，按重量降序排,重量相同时目的港降序
                    temp_list = [c for c in RealClist if c.setStack == column]
                    temp_list = sorted(temp_list, key=lambda c: (c.weight, c.Cp), reverse=True)
                    for i in range(numt):
                        try:
                            temp_list[i].setTier = i
                        except:
                            continue
                else:
                    temp_list = [c for c in RealClist if c.setStack == column]
                    temp_list = sorted(temp_list, key=lambda c: c.Cp, reverse=True)  # 甲板下，按目的港降序
                    for i in range(numt):
                        try:
                            temp_list[i].setTier = i
                        except:
                            continue
            return RealClist
        sol=Solution()
        for j in range(D):
            sol.X1.append(xs[j])    # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_=[]
        for i in range(D):
            Containers_.append(List_Containers_[sol.X1[i]])
        # 初始化子集列表
        num_subsets=int(numb/2)   # 分3个大贝位，甲板上下两个贝位为一个大贝位,待改？
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
            bun=int(b+numb/2)   # 甲板下
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
        # 箱分到箱位
        n=10    # 子问题迭代轮次
        sol.H1=0
        sol.H3=0
        # best_individual_list=[]
        for b in Vessel_Bays_:  # 对每个贝
            print('b',b.id)
            real_container_count = sum(1 for container in b.Container_Index if container.isReal)    # 分到贝位的真实箱数量，等于个体维度
            real_container_list = [container for container in b.Container_Index if container.isReal]    # 真实箱列表
            print('箱：',[c.id for c in real_container_list])
            subMde=sub_MODE(real_container_count,b,real_container_list)
            subMde.Pareto()
            for _ in range(n):
                subMde.Mutation()
                subMde.Cross()
                subMde.Select()
                subMde.Pareto()
            # sol.H1+=subMde.PopPareto[0].Fitness[0]  # 返回帕累托前沿的第一个个体，待改，返回曼哈顿距离最小的个体
            # sol.H3 += subMde.PopPareto[0].Fitness[1]
            # 找到帕累托前沿中距(0,0)曼哈顿距离最小的个体
            min_manhattan_distance = float('inf')
            best_individual = None
            for individual in subMde.PopPareto:
                manhattan_distance = abs(individual.Fitness[0]) + abs(individual.Fitness[1])
                if manhattan_distance < min_manhattan_distance:
                    min_manhattan_distance = manhattan_distance
                    best_individual = individual
            sol.H1 += best_individual.Fitness[0]
            print('best_ind:',best_individual.X)
            out_CList=outcome_sub_Decoding(best_individual.X,b,real_container_list)
            for c in out_CList:
                print('id',c.id)
                print('s',c.setStack)
                print('t',c.setTier)
            sol.H3 += best_individual.Fitness[1]
        Fitnessvalue=[sol.H1,sol.H2,sol.H3]
        return Fitnessvalue,Containers_

    def Decoding_H2(xs): # 输入个体前半（贝位分箱），输出f2(混装指标）
        sol=Solution()
        for j in range(D):
            sol.X1.append(xs[j])    # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_=[]
        for i in range(D):
            Containers_.append(List_Containers_[sol.X1[i]])
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

    class Individual:
        def __init__(self, d):
            self.D = d  # 维数
            self.Xs = [i for i in range(self.D)]  # Xs 是箱号列表，初始化为从 0 到 D-1 的顺序
            newClist = getNewContainersList(K_set)  # 获取新的箱列表（分组并打乱）
            # self.X = [container.id for container in newClist]  # 直接从 newClist 中取箱号

            self.Fitness = Decoding_Heuristic(self.Xs)  # 假设适应度计算基于 Xs
            self.RealFit = True  # 适应度是否为真实评估


    class sub_Individual:   # 用于每个贝位分箱位
        def __init__(self, d, min_value=0, max_value=int(nums-1), max_count=numt):
            self.D=d
            self.min_value = min_value
            self.max_value = max_value
            self.max_count = max_count
            self.X = self.initialize_vector()   # 个体
            self.repair_vector()
            self.Fitness=[] # 列重差求和，倒箱数     没法在实例化的适合计算适应度值，因为得传b和Clist

        def initialize_vector(self):
            # 初始化一个随机向量，取值范围在0到3之间的整数
            return np.random.randint(self.min_value, self.max_value + 1, self.D)

        def repair_vector(self):
            # 修复向量，确保每个列号的元素个数不超过5，且在0到3的范围内
            counts = np.zeros(self.max_value + 1, dtype=int)
            for i in range(self.D):
                value = self.X[i]
                if counts[value] >= self.max_count:
                    # 如果某列号超过了5次，就从当前列的下一列开始找不超过5次的列号
                    new_value = (value + 1) % (self.max_value + 1)
                    while counts[new_value] >= self.max_count:
                        new_value = (new_value + 1) % (self.max_value + 1)
                    self.X[i] = new_value
                counts[self.X[i]] += 1

    def sub_Decoding_pymoo(x,b,RealClist): # 箱位分箱解码,个体向量、预配列重、箱列表
        # print('xbefore',x)
        def repair_vector(x, max_value=int(nums - 1), max_count=numt):
            # x = np.array(x, dtype=int)  # 这是向下取整
            x = np.round(x).astype(int)
            counts = np.zeros(max_value + 1, dtype=int)
            for i in range(len(x)):
                value = x[i]
                if counts[value] >= max_count:
                    # If the column index appears too many times, assign a new valid index
                    new_value = (value + 1) % (max_value + 1)
                    while counts[new_value] >= max_count:
                        new_value = (new_value + 1) % (max_value + 1)
                    x[i] = new_value
                counts[x[i]] += 1
            return x

        # Repair the individual's vector to satisfy constraints before further processing
        x = repair_vector(x)
        # print('xafter',x)
        # w0四列，x匹配到箱重，作差求和
        column_weights = {i: 0 for i in range(nums)}  # 假设列号范围是0~3
        for i in range(len(x)):
            column = x[i]  # 取x向量中当前箱子的列号
            weight = RealClist[i].weight  # 取RealClist中对应箱子的重量
            RealClist[i].setStack=column    # 箱分配列号
            column_weights[column] += weight  # 将重量累加到对应的列号中
        H1_differences = 0
        H3_reload=0
        for column in range(nums):  # 列重差求和
            difference = (b.W0[column] - column_weights[column])**2
            H1_differences+=difference
            if b.Deck:  # 如果甲板以上，按重量降序排,重量相同时目的港降序
                temp_list=[c for c in RealClist if c.setStack==column]
                temp_list = sorted(temp_list, key=lambda c: (c.weight, c.Cp), reverse=True)
                for i in range(numt):
                    try:
                        temp_list[i].setTier=i
                    except:
                        continue
                for i, c1 in enumerate(temp_list[:-1]):
                    for c2 in temp_list[i + 1:]:
                        if c2.Cp>c1.Cp: # 因目的港的倒箱
                            H3_reload+=1
            else:
                temp_list=[c for c in RealClist if c.setStack==column]
                temp_list = sorted(temp_list, key=lambda c: c.Cp, reverse=True) # 甲板下，按目的港降序
                for i in range(numt):
                    try:
                        temp_list[i].setTier=i
                    except:
                        continue
        b.H1_Difference=H1_differences
        b.H3_reload=H3_reload
        return [H1_differences,H3_reload]

    def analyze_residuals(MDE,model_name='RandomForest'):
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        real_fitness_values = []
        predicted_fitness_values = []
        residuals = []

        for individual in MDE.PopInit:
            # 计算真实适应度
            real_fitness = Decoding_Heuristic(individual.Xs_)
            real_fitness_subset = [real_fitness[0], real_fitness[2]]
            real_fitness_values.append(real_fitness_subset)
            # real_fitness_values.append(real_fitness)
            # 使用代理模型预测适应度
            predicted_fitness = np.array([MDE.rf_models[j].predict([individual.X])[0] for j in range(2)])
            # predicted_fitness = np.array([
            #     MDE.rf_models[j].predict(scaler.transform([individual.X]).reshape(1, -1))[0]
            #     for j in range(2)
            # ])

            predicted_fitness_values.append(predicted_fitness)

            # # 计算残差
            # print('real:',real_fitness_subset)
            # print('pre:',predicted_fitness)
            residual = np.array(real_fitness_subset) - predicted_fitness
            residuals.append(residual)

        # 将残差和适应度值转换为numpy数组，以便进行后续分析
        residuals = np.array(residuals)
        real_fitness_values = np.array(real_fitness_values)
        predicted_fitness_values = np.array(predicted_fitness_values)

        # 绘制残差图
        plt.figure(figsize=(14, 7))

        for i, idx in enumerate([0, 2]):  # 0 表示第一维，2 表示第三维
            plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
            plt.scatter(range(len(residuals)), residuals[:, i], c='blue', marker='o')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f"Residuals for Fitness {idx + 1} ({model_name})")
            plt.xlabel("Individuals")
            plt.ylabel("Residual (Real - Predicted)")

        plt.tight_layout()  # 自动调整子图布局，避免重叠
        plt.show()

        # 打印残差统计信息
        mse = np.mean(residuals ** 2, axis=0)
        print(f"mse for {model_name}: {mse}")
        print(f"Max residuals for {model_name}: {np.max(residuals, axis=0)}")
        print(f"Min residuals for {model_name}: {np.min(residuals, axis=0)}")

        # 计算并打印每个适应度的 R² 值
        for i in range(2):
            r2 = r2_score(real_fitness_values[:, i], predicted_fitness_values[:, i])
            # adjusted_r2 = adjusted_r2_score(r2, len(real_fitness_values), num_slot)
            print(f"R² for Fitness {i + 1} ({model_name}): {r2}")

    class SubProblem(Problem):
        def __init__(self, num_variables, bay, container_list):
            super().__init__(n_var=num_variables, n_obj=2, xl=0, xu=int(nums - 1))
            self.bay = bay
            self.container_list = container_list

        def _evaluate(self, X, out, *args, **kwargs):
            fitness_values = [sub_Decoding_pymoo(x, self.bay, self.container_list) for x in X]
            out["F"] = np.array(fitness_values)
    class SequenceSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            n_var = problem.n_var
            # 随机生成 n_samples 个无重复整数序列
            samples = np.array([np.random.permutation(n_var) for _ in range(n_samples)])
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
    # nsga2
    def Decoding_Heuristic2(xs):  # 输入个体前半（贝位分箱），输出f2(混装指标）、贝位分到的箱
        sol = Solution()
        for j in range(D):
            sol.X1.append(xs[j])  # xs应该是启发式分箱顺序的箱号
        # print('sol.X1',sol.X1)
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_ = []
        for i in range(D):
            Containers_.append(List_Containers_[int(sol.X1[i])])
        # 初始化子集列表
        num_subsets = int(numb / 2)  # 分3个大贝位，甲板上下两个贝位为一个大贝位
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

        # 评价贝位分配指标
        for id, b in enumerate(Vessel_Bays_):
            # 统计船舶贝位内集装箱的目的港和堆场贝位数
            b.H1_Diversity = len(b.Destinations) + len(b.YardBlocks)

        # 个体适应值更新
        sol.H2 = 0
        for i, _ in enumerate(Vessel_Bays_):
            sol.H2 += Vessel_Bays_[i].H1_Diversity
        # 箱分到箱位
        n = 10  # 子问题迭代轮次
        sol.H1 = 0
        sol.H3 = 0
        # best_individual_list=[]
        for b in Vessel_Bays_:
            real_container_list = [container for container in b.Container_Index if container.isReal]
            sub_problem = SubProblem(len(real_container_list), b, real_container_list)

            # 执行NSGA2算法解决子问题
            algorithm = NSGA2(pop_size=sub_np)
            res = minimize(sub_problem, algorithm, ('n_gen', 10), seed=random.randint(0,10), verbose=False)

            # 选择帕累托前沿中最优解（如距离最小的解）
            best_solution = min(res.F, key=lambda f: abs(f[0]) + abs(f[1]))
            sol.H1 += best_solution[0]
            sol.H3 += best_solution[1]
        Fitnessvalue = [sol.H1, sol.H2, sol.H3]
        # print(Fitnessvalue)
        return Fitnessvalue


    class upLevelProblem2(Problem):
        def __init__(self, n_var):
            xl = np.zeros(n_var)  # 下限是0，长度为n_var
            xu = np.full(n_var, n_var - 1)  # 上限是n_var-1，长度为n_var
            super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu,type_var=Integer)

        def _evaluate(self, X, out, *args, **kwargs):
            results = [Decoding_Heuristic2(x) for x in X]
            out["F"] = np.array(results)

    #nsga3
    def Decoding_Heuristic3(xs):  # 输入个体前半（贝位分箱），输出f2(混装指标）、贝位分到的箱
        sol = Solution()
        for j in range(D):
            sol.X1.append(xs[j])  # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_ = []
        for i in range(D):
            Containers_.append(List_Containers_[int(sol.X1[i])])
        # 初始化子集列表
        num_subsets = int(numb / 2)  # 分3个大贝位，甲板上下两个贝位为一个大贝位
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

        # 评价贝位分配指标
        for id, b in enumerate(Vessel_Bays_):
            # 统计船舶贝位内集装箱的目的港和堆场贝位数
            b.H1_Diversity = len(b.Destinations) + len(b.YardBlocks)

        # 个体适应值更新
        sol.H2 = 0
        for i, _ in enumerate(Vessel_Bays_):
            sol.H2 += Vessel_Bays_[i].H1_Diversity
        # 箱分到箱位
        n = 10  # 子问题迭代轮次
        sol.H1 = 0
        sol.H3 = 0
        # best_individual_list=[]
        for b in Vessel_Bays_:
            real_container_list = [container for container in b.Container_Index if container.isReal]
            sub_problem = SubProblem(len(real_container_list), b, real_container_list)
            ref_dirs_new = get_reference_directions("das-dennis", 2, n_partitions=12)
            algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs_new)
            res = minimize(sub_problem, algorithm, ('n_gen', 10), seed=random.randint(0,10), verbose=False)
            # 选择帕累托前沿中最优解（如距离最小的解）
            best_solution = min(res.F, key=lambda f: abs(f[0]) + abs(f[1]))
            sol.H1 += best_solution[0]
            sol.H3 += best_solution[1]
        Fitnessvalue = [sol.H1, sol.H2, sol.H3]
        # print(Fitnessvalue)
        return Fitnessvalue
    # 定义多目标问题
    class upLevelProblem3(Problem):
        def __init__(self, n_var):
            xl = np.zeros(n_var)  # 下限是0，长度为n_var
            xu = np.full(n_var, n_var - 1)  # 上限是n_var-1，长度为n_var
            super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu,type_var=Integer)

        def _evaluate(self, X, out, *args, **kwargs):
            results = [Decoding_Heuristic3(x) for x in X]
            out["F"] = np.array(results)
    #moead
    def Decoding_Heuristic4(xs):  # 输入个体前半（贝位分箱），输出f2(混装指标）、贝位分到的箱
        sol = Solution()
        for j in range(D):
            sol.X1.append(xs[j])  # xs应该是启发式分箱顺序的箱号
        Vessel_Bays_ = sol.Vessel_Bays_
        List_Containers_ = sol.List_Containers_
        Containers_ = []
        for i in range(D):
            Containers_.append(List_Containers_[int(sol.X1[i])])
        # 初始化子集列表
        num_subsets = int(numb / 2)  # 分3个大贝位，甲板上下两个贝位为一个大贝位
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

        # 评价贝位分配指标
        for id, b in enumerate(Vessel_Bays_):
            # 统计船舶贝位内集装箱的目的港和堆场贝位数
            b.H1_Diversity = len(b.Destinations) + len(b.YardBlocks)

        # 个体适应值更新
        sol.H2 = 0
        for i, _ in enumerate(Vessel_Bays_):
            sol.H2 += Vessel_Bays_[i].H1_Diversity
        # 箱分到箱位
        n = 10  # 子问题迭代轮次
        sol.H1 = 0
        sol.H3 = 0
        # best_individual_list=[]
        for b in Vessel_Bays_:
            real_container_list = [container for container in b.Container_Index if container.isReal]
            sub_problem = SubProblem(len(real_container_list), b, real_container_list)
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=50,  # 邻居数
                prob_neighbor_mating=0.9,  # 选择邻居交配的概率
            )
            res = minimize(sub_problem, algorithm, ('n_gen', 10), seed=random.randint(0,10), verbose=False)
            # 选择帕累托前沿中最优解（如距离最小的解）
            best_solution = min(res.F, key=lambda f: abs(f[0]) + abs(f[1]))
            sol.H1 += best_solution[0]
            sol.H3 += best_solution[1]
        Fitnessvalue = [sol.H1, sol.H2, sol.H3]
        # print(Fitnessvalue)
        return Fitnessvalue
    # 定义多目标问题
    class upLevelProblem4(Problem):
        def __init__(self, n_var):
            xl = np.zeros(n_var)  # 下限是0，长度为n_var
            xu = np.full(n_var, n_var - 1)  # 上限是n_var-1，长度为n_var
            super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu,type_var=Integer)

        def _evaluate(self, X, out, *args, **kwargs):
            results = [Decoding_Heuristic4(x) for x in X]
            out["F"] = np.array(results)
    mode_aver_y=[]
    mode_aver_time=[]
    mode_aver_f1=[]
    mode_aver_f2=[]
    mode_aver_f3=[]
    mode_aver_hv=[]
    mode_aver_igd=[]
    mode_aver_len_pareto=[]
    mode_sumset=[]
    nsga2_sumset=[]
    nsga2_aver_f1=[]
    nsga2_aver_f2=[]
    nsga2_aver_f3=[]
    nsga2_aver_len_pareto=[]
    nsga3_sumset=[]
    nsga3_aver_f1=[]
    nsga3_aver_f2=[]
    nsga3_aver_f3=[]
    nsga3_aver_len_pareto=[]
    # moead
    # 创建问题实例
    moead_sumset=[]
    moead_aver_f1=[]
    moead_aver_f2=[]
    moead_aver_f3=[]
    moead_aver_len_pareto=[]
    test_num=3   # 求平均
    N = 10  # 算法迭代次数
    print(time.time())
    for test in range(test_num):
        #MODE
        print('---------------------------------MODE_Start------------------------------')
        startTime = time.time()
        MDE = MODE(D)
        # DB 档案，存用于训练模型的个体，时间窗更新
        DB=[]
        DB_maxlen=int(NP*4) # DB最大容量
        for ind in range(NP):   # 先存全部的PopInit个体
            DB.append(copy.deepcopy(MDE.PopInit[ind]))
        MDE.Pareto()
        # 用于记录每一代的帕累托前沿的适应度值
        # fitness1_history = []
        # fitness2_history = []
        # fitness3_history = []
        fitness_sum_history = []
        for I in range(N):
            # pass
            MDE.Mutation()
            MDE.Cross()
            MDE.Select()
            MDE.Pareto()
            print('len',len(MDE.PopPareto))
            # print('lenDB',len(DB))
            if I>=1:    # 更新存档
                update_DB([copy.deepcopy(ind) for ind in MDE.PopInit])

            # 记录每一代帕累托前沿适应度值的平均值——改成记求和最小
            if len(MDE.PopPareto) > 0:
                fitness_sums = [ind.Fitness[0] + ind.Fitness[1] + ind.Fitness[2] for ind in MDE.PopPareto]
                min_fitness_sum = min(fitness_sums)
                min_index = fitness_sums.index(min_fitness_sum)
                min_individual = MDE.PopPareto[min_index]   # 帕累托集中被选中的个体
                ind_fitness=min_individual.Fitness
                # fitness1_history.append(ind_fitness[0])
                # fitness2_history.append(ind_fitness[1])
                # fitness3_history.append(ind_fitness[2])
                fitness_sum_history.append(min_fitness_sum)

            if I ==  0:
                print('i=', I)
                max_fitness_sum = max(fitness_sums)
                print('min:', min_fitness_sum)
                print('max:', max_fitness_sum)
            elif I == N - 1:
                print('i=', I)
                max_fitness_sum = max(fitness_sums)
                print('min:', min_fitness_sum)
                print('max:', max_fitness_sum)
                print('min_f1:',ind_fitness[0])
                print('min_f2:',ind_fitness[1])
                print('min_f3:',ind_fitness[2])

                mode_aver_y.append(min_fitness_sum)
                mode_aver_f1.append(ind_fitness[0])
                mode_aver_f2.append(ind_fitness[1])
                mode_aver_f3.append(ind_fitness[2])
                mode_aver_len_pareto.append(len(MDE.PopPareto))
                mode_pareto=[]
                for ind in MDE.PopPareto:
                    mode_pareto.append(ind.Fitness)
                    mode_sumset.append(ind.Fitness)
                    print('fit',ind.Fitness)

                # print('Xs:', min_individual.Xs_)
                # Fit,Clist=outcome_Decoding_Heuristic(min_individual.Xs_)
                # print('fitness',Fit)

        endTime = time.time()
        deltaTime = endTime - startTime
        print("MODE用时：", deltaTime)
        mode_aver_time.append(deltaTime)

        print('---------------------------------MODE_End------------------------------')

        #nsga2
        print('--------------------------------nsga2-start----------------------------------')
        problem = upLevelProblem2(n_var=D)
        start_time=time.time()
        algorithm = NSGA2(
            pop_size=100,
            sampling=SequenceSampling(),
            repair=SequenceRepairWithRound()
        )        # 自定义的 sampling, crossover, mutation

        res = minimize(problem,
                       algorithm,
                       ('n_gen', N),
                       seed=random.randint(0,10),
                       verbose=True)

        print("最终得到的解:")
        nsga2_solutions = np.unique(res.F, axis=0)  # 剔除重复解
        fitness_sums = [f[0] + f[1] + f[2] for f in nsga2_solutions]
        min_fitness_sum = min(fitness_sums)
        min_index = fitness_sums.index(min_fitness_sum)
        nsga2_aver_f1.append(nsga2_solutions[min_index][0])
        nsga2_aver_f2.append(nsga2_solutions[min_index][1])
        nsga2_aver_f3.append(nsga2_solutions[min_index][2])
        nsga2_aver_len_pareto.append(len(nsga2_solutions))
        print(nsga2_solutions)
        for f in nsga2_solutions:
            nsga2_sumset.append(f)
        end_time=time.time()
        print('nsga2time:',end_time-start_time)
        print('--------------------------------nsga2-end----------------------------------')

        # nsga3
        print('----------------------------nsga3-start------------------------')
        problem = upLevelProblem3(n_var=D)
        start_time=time.time()
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        algorithm = NSGA3(pop_size=100,
                          ref_dirs=ref_dirs,
                          sampling=SequenceSampling(),
                          repair=SequenceRepairWithRound())
        # 优化问题
        res = minimize(problem,
                       algorithm,
                       ('n_gen', N),
                       seed=random.randint(0,10),
                       verbose=True)
        print("最终得到的解:")

        nsga3_solutions = np.unique(res.F, axis=0)  # 剔除重复解
        fitness_sums = [f[0] + f[1] + f[2] for f in nsga3_solutions]
        min_fitness_sum = min(fitness_sums)
        min_index = fitness_sums.index(min_fitness_sum)
        nsga3_aver_f1.append(nsga3_solutions[min_index][0])
        nsga3_aver_f2.append(nsga3_solutions[min_index][1])
        nsga3_aver_f3.append(nsga3_solutions[min_index][2])
        nsga3_aver_len_pareto.append(len(nsga3_solutions))
        print(nsga3_solutions)
        for f in nsga3_solutions:
            nsga3_sumset.append(f)
        end_time=time.time()
        print('nsga3time:',end_time-start_time)
        print('---------------------------------nsga3-End-------------------------')
        print('-------------------------moead-start--------------------------------')
        problem = upLevelProblem4(n_var=D)
        start_time=time.time()
        ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=12)
        # 设置MOEA/D算法参数
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=100,  # 邻居数
            prob_neighbor_mating=0.9,  # 选择邻居交配的概率
            sampling=SequenceSampling(),
            repair=SequenceRepairWithRound()

        )
        # 优化问题
        res = minimize(problem,
                       algorithm,
                       ('n_gen', N),
                       seed=random.randint(0,10),
                       verbose=True)
        print("最终得到的解:")

        moead_solutions = np.unique(res.F, axis=0)  # 剔除重复解
        fitness_sums = [f[0] + f[1] + f[2] for f in moead_solutions]
        min_fitness_sum = min(fitness_sums)
        min_index = fitness_sums.index(min_fitness_sum)
        moead_aver_f1.append(moead_solutions[min_index][0])
        moead_aver_f2.append(moead_solutions[min_index][1])
        moead_aver_f3.append(moead_solutions[min_index][2])
        moead_aver_len_pareto.append(len(moead_solutions))
        print(moead_solutions)
        for f in moead_solutions:
            moead_sumset.append(f)
        end_time=time.time()
        print('moeadtime:',end_time-start_time)
        print('------------------------moead-end------------------------------------')
        if test==test_num-1:
            print('---------------------------------MODE_Start------------------------------')
            temp_y=sum(mode_aver_y)/len(mode_aver_y)
            temp_time=sum(mode_aver_time)/len(mode_aver_time)
            temp1=sum(mode_aver_f1)/len(mode_aver_f1)
            temp2=sum(mode_aver_f2)/len(mode_aver_f2)
            temp3=sum(mode_aver_f3)/len(mode_aver_f3)
            temp6=sum(mode_aver_len_pareto)/len(mode_aver_len_pareto)
            print('y:',mode_aver_y)
            print('time:',mode_aver_time)
            print('aver_y:',temp_y)
            print('aver_time:',temp_time)
            print('averf1:',temp1,'averf2:',temp2,'averf3:',temp3)
            print('aver_len_pareto:',temp6)
            print('---------------------------------MODE_End------------------------------')
            print('------------------------------nsga2-start---------------------------------')

            temp1 = sum(nsga2_aver_f1) / len(nsga2_aver_f1)
            temp2 = sum(nsga2_aver_f2) / len(nsga2_aver_f2)
            temp3 = sum(nsga2_aver_f3) / len(nsga2_aver_f3)
            temp6 = sum(nsga2_aver_len_pareto) / len(nsga2_aver_len_pareto)
            print('averf1:', temp1, 'averf2:', temp2, 'averf3:', temp3)
            print('aver_len_pareto:', temp6)
            print('---------------------------------------nsga2---end----------------------------------')
            print('----------------------------------nsga3-start------------------------------')
            temp1 = sum(nsga3_aver_f1) / len(nsga3_aver_f1)
            temp2 = sum(nsga3_aver_f2) / len(nsga3_aver_f2)
            temp3 = sum(nsga3_aver_f3) / len(nsga3_aver_f3)
            temp6 = sum(nsga3_aver_len_pareto) / len(nsga3_aver_len_pareto)
            print('averf1:', temp1, 'averf2:', temp2, 'averf3:', temp3)
            print('aver_len_pareto:', temp6)
            print('----------------------------------nsga3-end--------------------------------')
            print('-------------------------------moead-start------------------------')
            temp1 = sum(moead_aver_f1) / len(moead_aver_f1)
            temp2 = sum(moead_aver_f2) / len(moead_aver_f2)
            temp3 = sum(moead_aver_f3) / len(moead_aver_f3)
            temp6 = sum(moead_aver_len_pareto) / len(moead_aver_len_pareto)
            print('averf1:', temp1, 'averf2:', temp2, 'averf3:', temp3)
            print('aver_len_pareto:', temp6)
            print('--------------------------------moead-end-----------------------')
            # 计算指标
            def is_dominated(sol, solutions):
                # 保证前沿上解互不支配
                return np.any(np.all(solutions <= sol, axis=1) & np.any(solutions < sol, axis=1))
            # 去除重复解
            mode_sumset = np.unique(mode_sumset, axis=0)
            nsga2_sumset = np.unique(nsga2_sumset, axis=0)
            nsga3_sumset = np.unique(nsga3_sumset, axis=0)
            moead_sumset = np.unique(moead_sumset, axis=0)

            # 合并 mode, nsga2, nsga3 和 moead 的非支配解到 data
            data = np.vstack((mode_sumset, nsga2_sumset, nsga3_sumset, moead_sumset))
            print('data',data)
            unique_solutions = np.unique(data, axis=0)

            # 计算 data 的非支配前沿
            pareto_front = []
            for sol in unique_solutions:
                if not is_dominated(sol, unique_solutions):
                    pareto_front.append(sol)
            pareto_front = np.array(pareto_front)
            print('pareto_front:', pareto_front)
            # 计算mode的非支配前沿
            mode_front = []
            for sol in mode_sumset:
                if not is_dominated(sol, mode_sumset):
                    mode_front.append(sol)
            mode_front = np.array(mode_front)
            print('mode_front:', mode_front)

            # 计算nsga2的非支配前沿
            nsga2_front = []
            for sol in nsga2_sumset:
                if not is_dominated(sol, nsga2_sumset):
                    nsga2_front.append(sol)
            nsga2_front = np.array(nsga2_front)
            print('nsga2_front:', nsga2_front)
            # 计算 NSGA-III 的非支配前沿
            nsga3_front = []
            for sol in nsga3_sumset:
                if not is_dominated(sol, nsga3_sumset):
                    nsga3_front.append(sol)
            nsga3_front = np.array(nsga3_front)
            print('nsga3_front:', nsga3_front)

            # 计算 MOEA/D 的非支配前沿
            moead_front = []
            for sol in moead_sumset:
                if not is_dominated(sol, moead_sumset):
                    moead_front.append(sol)
            moead_front = np.array(moead_front)
            print('moead_front:', moead_front)

            # 设置参考点
            reference_point = [1.1, 1.1, 1.1]
            print('参考点：', reference_point)

            # 计算 mode 的 HV 和 IGD
            mode_hv_value = calculate_hv(mode_front, reference_point,data)
            # print('mode_front',mode_front)
            mode_igd_value = calculate_igd(mode_front, pareto_front,data)
            print('modehv:', mode_hv_value)
            print('modeigd:', mode_igd_value)

            # 计算 nsga2 的 HV 和 IGD
            nsga2_hv_value = calculate_hv(nsga2_front, reference_point,data)
            nsga2_igd_value = calculate_igd(nsga2_front, pareto_front,data)
            print('nsga2hv:', nsga2_hv_value)
            print('nsga2igd:', nsga2_igd_value)

            # 计算 nsga3 的 HV 和 IGD
            nsga3_hv_value = calculate_hv(nsga3_front, reference_point,data)
            nsga3_igd_value = calculate_igd(nsga3_front, pareto_front,data)
            print('nsga3hv:', nsga3_hv_value)
            print('nsga3igd:', nsga3_igd_value)

            # 计算 moead 的 HV 和 IGD
            moead_hv_value = calculate_hv(moead_front, reference_point,data)
            moead_igd_value = calculate_igd(moead_front, pareto_front,data)
            print('moeadhv:', moead_hv_value)
            print('moeadigd:', moead_igd_value)
