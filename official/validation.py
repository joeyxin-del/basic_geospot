# Validation and Scoring tool for the spotGEO competition on kelvins.esa.int.
# 用于kelvins.esa.int上spotGEO竞赛的验证和评分工具

# Imports
# 导入必要的库
import numpy as np  # 导入numpy库用于数值计算
import json  # 导入json库用于处理JSON数据
import jsonschema  # 导入jsonschema库用于JSON模式验证
import sys  # 导入sys库用于系统相关操作

from scipy.spatial.distance import cdist  # 从scipy导入cdist用于计算距离矩阵
from scipy.optimize import linear_sum_assignment  # 从scipy导入线性分配算法
from collections import defaultdict  # 导入defaultdict用于创建默认字典
from zipfile import ZipFile  # 导入ZipFile用于处理zip文件

# Constants
# 定义常量
min_seq_number = 1  # 最小序列号
max_seq_number = 5120  # 最大序列号
frames_per_sequence = 5  # 每个序列的帧数
img_width  = 639.5  # 图像宽度
img_height = 479.5  # 图像高度
min_size = -0.5  # 最小尺寸
max_number_of_objects = 30  # 最大对象数量


# Submissions have to follow the following schema
# 定义提交文件必须遵循的JSON模式
schema = {
    "type": "array",  # 类型为数组
    "items": {
        "type": "object",  # 数组中的每个元素都是对象
        "properties": {
            "sequence_id": { "type" : "integer",  # 序列ID必须是整数
                             "minimum": min_seq_number,  # 最小值为min_seq_number
                             "maximum": max_seq_number},  # 最大值为max_seq_number
            "frame":       { "type" : "integer",  # 帧号必须是整数
                             "minimum": 1,  # 最小值为1
                             "maximum": frames_per_sequence},  # 最大值为frames_per_sequence
            "num_objects": { "type" : "integer",  # 对象数量必须是整数
                             "minimum": 0,  # 最小值为0
                             "maximum": max_number_of_objects},  # 最大值为max_number_of_objects
            "object_coords": { "type" : "array",  # 对象坐标必须是数组
                               "items": {
                                   "type": "array",  # 数组中的每个元素也是数组
                                   "items": [ {"type": "number",  # 第一个元素是数字
                                               "minimum": min_size,  # 最小值为min_size
                                               "maximum": img_width },  # 最大值为img_width
                                              {"type": "number",  # 第二个元素是数字
                                               "minimum": min_size,  # 最小值为min_size
                                               "maximum": img_height } ]  # 最大值为img_height
                                   }
                             }
        },
        "required": ["sequence_id", "frame", "num_objects", "object_coords"]  # 必需字段
    },
}

# Helper functions
# 辅助函数
def flat_to_hierarchical(labels):
    """ Transforms a flat array of json-objects to a hierarchical python dict, indexed by
        sequence number and frame id. """
    # 将扁平的JSON对象数组转换为按序列号和帧ID索引的层次化Python字典
    seqs = dict()  # 创建空字典
    for label in labels:  # 遍历每个标签
        seq_id = label['sequence_id']  # 获取序列ID
        frame_id = label['frame']  # 获取帧ID
        coords = label['object_coords']  # 获取对象坐标
        
        if seq_id not in seqs.keys():  # 如果序列ID不在字典中
            seqs[seq_id] = defaultdict(dict)  # 创建新的默认字典
        seqs[seq_id][frame_id] = np.array(coords)  # 将坐标转换为numpy数组并存储
    
    return seqs  # 返回转换后的字典


def score_frame(X, Y, tau=10, eps=3):
    """ Scoring Prediction X on ground-truth Y by linear assignment. """
    # 使用线性分配算法对预测X和真实值Y进行评分
    if len(X) == 0 and len(Y) == 0:  # 如果没有预测和真实值
        # no objects, no predictions means perfect score
        # 没有对象和预测意味着完美分数
        TP, FN, FP, sse = 0, 0, 0, 0  # 初始化所有指标为0
    elif len(X) == 0 and len(Y) > 0:  # 如果没有预测但有真实值
        # no predictions but objects means false negatives
        # 没有预测但有对象意味着假阴性
        TP, FN, FP, sse = 0, len(Y), 0, len(Y) * tau**2  # 计算假阴性和误差
    elif len(X) > 0 and len(Y) == 0:  # 如果有预测但没有真实值
        # predictions but no objects means false positives
        # 有预测但没有对象意味着假阳性
        TP, FN, FP, sse = 0, 0, len(X), len(X) * tau**2  # 计算假阳性和误差
    else:  # 如果既有预测又有真实值
        # compute Euclidean distances between prediction and ground truth
        # 计算预测和真实值之间的欧几里得距离
        D = cdist(X, Y)  # 计算距离矩阵

        # truncate distances that violate the threshold
        # 截断超过阈值的距离
        D[D > tau] = 1000  # 将超过阈值的距离设为1000

        # compute matching by solving linear assignment problem
        # 通过解决线性分配问题计算匹配
        row_ind, col_ind = linear_sum_assignment(D)  # 使用匈牙利算法进行匹配
        matching = D[row_ind, col_ind]  # 获取匹配的距离

        # true positives are matches within the threshold
        # 真阳性是阈值内的匹配
        TP = sum(matching <= tau)  # 计算真阳性数量

        # false negatives are missed ground truth points or matchings that violate the threshold
        # 假阴性是未匹配的真实点或超过阈值的匹配
        FN = len(Y) - len(row_ind) + sum(matching > tau)  # 计算假阴性数量

        # false positives are missing predictions or matchings that violate the threshold
        # 假阳性是未匹配的预测或超过阈值的匹配
        FP = len(X) - len(row_ind) + sum(matching > tau)  # 计算假阳性数量
        
        # compute truncated regression error
        # 计算截断回归误差
        tp_distances = matching[matching < tau]  # 获取阈值内的距离
        # truncation
        # 截断
        tp_distances[tp_distances < eps] = 0  # 将小于eps的距离设为0
        # squared error with constant punishment for false negatives and true positives
        # 计算平方误差，对假阴性和假阳性使用常数惩罚
        sse = sum(tp_distances) + (FN + FP) * tau**2  # 计算总平方误差
    
    return TP, FN, FP, sse  # 返回所有指标
    

def score_sequence(X, Y, tau=10, eps=3):
    # check that X and Y cover all 5 frames
    # 检查X和Y是否覆盖所有5帧
    assert set(X.keys()) == set(Y.keys())  # 确保X和Y的键相同
    
    frame_scores = [score_frame(X[k], Y[k], tau=tau, eps=eps) for k in X.keys()]  # 计算每帧的分数
    TP = sum([x[0] for x in frame_scores])  # 计算总真阳性
    FN = sum([x[1] for x in frame_scores])  # 计算总假阴性
    FP = sum([x[2] for x in frame_scores])  # 计算总假阳性
    sse = sum([x[3] for x in frame_scores])  # 计算总平方误差
    
    mse = 0 if (TP + FN + FP) == 0 else sse / (TP + FN + FP)  # 计算均方误差
    return TP, FN, FP, mse  # 返回所有指标


def score_sequences(X, Y, tau=10, eps=3, taboolist=[]):
    """ scores a complete submission except sequence_ids that are listed
        in the taboolist. """
    # 对完整提交进行评分，除了taboolist中列出的sequence_ids
    # check that each sequence has been predicted
    # 检查每个序列是否都被预测
    assert set(X.keys()) == set(Y.keys())  # 确保X和Y的键相同
    
    # we filter the identifiers from the taboolist
    # 从taboolist中过滤标识符
    identifiers = set(X.keys()) - set(taboolist)  # 获取不在taboolist中的标识符
    
    # compute individual sequence scores
    # 计算每个序列的分数
    seq_scores = [score_sequence(X[k], Y[k], tau=tau, eps=eps) for k in identifiers]  # 计算每个序列的分数
    TP = sum([x[0] for x in seq_scores])  # 计算总真阳性
    FN = sum([x[1] for x in seq_scores])  # 计算总假阴性
    FP = sum([x[2] for x in seq_scores])  # 计算总假阳性
    mse = sum([x[3] for x in seq_scores])  # 计算总均方误差
    
    precision = TP / (TP + FP)  # 计算精确率
    recall = TP / (TP + FN)  # 计算召回率
    F1 = 2 * precision * recall / (precision + recall)  # 计算F1分数
    
    return precision, recall, F1, mse  # 返回所有指标


def compute_score(predictions, labels):
    """ Scores a submission `predictions` against ground-truth `labels`. Does
    not perform any validation and expects `predictions` and `labels` to be
    valid paths to .json-files. """
    # 对提交的predictions与真实值labels进行评分，不进行验证，期望predictions和labels是有效的.json文件路径
    with open(predictions, 'rt') as fp:  # 打开预测文件
        predictions_h = flat_to_hierarchical(json.load(fp))  # 加载并转换预测数据
    
    with open(labels, 'rt') as fp:  # 打开标签文件
        labels_h = flat_to_hierarchical(json.load(fp))  # 加载并转换标签数据
    
    precision, recall, F1, mse = score_sequences(predictions_h, labels_h)  # 计算分数
    
    return (1 - F1, mse)  # 返回(1-F1)和均方误差
    

def validate_json(labels):
    """ Valides whether `labels` follow the required formats to be accepted
        for computing a score. """
    # 验证labels是否符合计算分数所需的格式要求
    # 1. Check whether the json follows correct input formats
    # 检查JSON是否符合正确的输入格式
    jsonschema.validate(labels, schema)  # 使用jsonschema验证格式
    
    # 2. jsonschema is not powerful enough to appropriately check for duplicates
    # jsonschema不足以检查重复项
    identifiers = [(label['sequence_id'], label['frame']) for label in labels]  # 获取所有标识符
    if not len(set(identifiers)) == len(identifiers):  # 检查是否有重复
        raise ValueError('Error. You have duplicates in your submission. Make sure each combination of sequence_id and frame is unique.')  # 抛出重复错误
        
    # 3. We need an identifier for each sequence and frame combination
    # 我们需要每个序列和帧组合的标识符
    needed_identifiers = {(i,j) for i in range(min_seq_number, max_seq_number + 1) for j in range(1, frames_per_sequence + 1)}  # 生成所需的所有标识符
    missing_identifiers = needed_identifiers - set(identifiers)  # 找出缺失的标识符
    
    if len(missing_identifiers) > 0:  # 如果有缺失的标识符
        raise ValueError('Error. Your submission needs to predict the following sequence_ids and frames: {}'.format(missing_identifiers))  # 抛出缺失错误
        
    # 4. Make sure the number of predicted objects corresponds to the correct array dimensions
    # 确保预测的对象数量与正确的数组维度相对应
    for label in labels:  # 遍历每个标签
        if len(label['object_coords']) != label['num_objects']:  # 检查坐标数量是否与对象数量匹配
            raise ValueError('Error. You indicated to predict {:d} objects, but give coordinates for {:d} in sequence {:d}, frame {:d}.'.format(label['num_objects'], len(label['object_coords']), label['sequence_id'], label['frame']))  # 抛出不匹配错误

    # The file validates successfully
    # 文件验证成功
    return True  # 返回True


# This script can run from the command line: python my_anno.json true_labels.json
# 这个脚本可以从命令行运行：python my_anno.json true_labels.json
if __name__ == '__main__':  # 如果是主程序
    if len(sys.argv)  not in [2, 3]:  # 如果参数数量不正确
        print('Usage: \n\tValidation: python my_anno.json\n\tscoring: python my_anno.json true_labels.json')  # 打印使用说明
    else:  # 如果参数数量正确
        print('Validating... ', end='', flush=True)  # 打印验证信息
        with open(sys.argv[1], 'rt') as fp:  # 打开第一个参数指定的文件
            jsonfile = json.load(fp)  # 加载JSON文件
            valid = validate_json(jsonfile)  # 验证JSON文件
        
        if valid:  # 如果验证通过
            print('passed!')  # 打印通过信息
            
        if len(sys.argv) == 3:  # 如果有第三个参数
            print('Compute score... ', end='', flush=True)  # 打印计算分数信息
            score, mse = compute_score(sys.argv[1], sys.argv[2])  # 计算分数
            print('Score: {:0.6f}, (MSE: {:0.6f})'.format(score, mse))  # 打印分数和均方误差