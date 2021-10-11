# -*- coding: utf-8 -*-
"""
#优化方法：对于长文本（大于100个字符），只进行单关键词匹配
Created on Fri Aug 27 14:49:45 2021
@author: admin
"""

import jieba
import time
import pandas as pd
import ahocorasick
from collections import Counter
import copy
from file_op import Txt_Op, sets_operation_2, sets_operation_3
import re

class AhocorasickNer_content:
    def __init__(self):
        self.actree = ahocorasick.Automaton()
    
    def del_stopwords(self, stopwords_list, sentence):    #去除文本中的停用词
        new_sentence = ''
        seg_list = list(jieba.cut(sentence, cut_all=False))
        #print(seg_list)
        for char in seg_list:
            if char in stopwords_list: #如果当前词汇在停用词库中，则用'*'替代
                char = '*'
            new_sentence += char
        return new_sentence
    
    # 添加文件中的关键词
    def add_keywords(self, user_dict_path, sen_type, custom_sensitive_words: list):
        flag = 0
        dcit = dict()
        force_dict = dict()
        if custom_sensitive_words:
            for word in custom_sensitive_words:
                word, flag = word.strip(), flag + 1
                self.actree.add_word(word, (flag, word))
                dcit[word] = '待指明'
        else:
            target_sensitive = pd.read_excel(user_dict_path, dtype=str).fillna('')
            target_sensitive_one_sen = target_sensitive[target_sensitive['target_sensitive_type']==sen_type]#
            force_sensitive = target_sensitive_one_sen[target_sensitive_one_sen['judge']=='1']
            force_dict = dict(zip(force_sensitive['sensitive_words'], force_sensitive['judge']))
            for i in range(len(target_sensitive_one_sen)):
                word, flag = target_sensitive_one_sen.iloc[i]['sensitive_words'].strip(), flag + 1
                if word:
                    self.actree.add_word(word, (flag, word))
                    dcit[word] = target_sensitive_one_sen.iloc[i]['target_sensitive_type']
        self.actree.make_automaton()
        self.dcit = dcit  # 创建关键字与其类别对应的键值对字典
        self.force_dict = force_dict # 创建关键字与强制匹配对应的字典
    
    def small_fun_old(self, detail, key, detail_ll, res):
        tmp = dict()
        description = []
        value = detail[key]
        for k, v in value.items():
            tmp_ = dict()
            tmp_['count'] = v
            tmp_['keyword'] = k
            description.append(tmp_)
        tmp['description'] = description
        tmp['type'] = key
        detail_ll.append(tmp)
        res['detail'] = detail_ll
        res['judge'] = '敏感'
        return res

    def small_fun(self, detail, key, detail_ll, res):
        #print(detail, key, detail_ll, res)
        tmp = {}
        description = []
        value = detail[key]
        for k, v in value.items():
            tmp_ = {'count': v, 'keyword': k}
            description.append(tmp_)
            res['judge'] = '敏感'
        if res['judge'] == '敏感':
            tmp['description'] = description
            tmp['type'] = key
            if tmp not in detail_ll:
                detail_ll.append(tmp)
            res['detail'] = detail_ll
            type_ll = []
            for i in range(len(detail_ll)):
                tmp2 = detail_ll[i]['type']
                type_ll.append(tmp2)
            type_ll = list(set(','.join(type_ll).split(',')))
            res['type'] = type_ll
        #print(res)
        return res
                
    def sovle_res_old(self, detail, res, factor_length):
        detail_keys = detail.keys()
        if '非敏感' not in detail_keys:
            detail_ll = []
            #print(factor_length)
            for key in detail_keys:
                if len(detail[key].keys()) >= factor_length:
                    res = self.small_fun(detail, key, detail_ll, res)
                detail_ll = []
                if res['judge'] == '非敏感':
                    detail_tmp = copy.deepcopy(detail)
                    for key in detail_keys:
                        for word in list(detail[key].keys()):
                            if word not in self.force_dict:
                                del detail_tmp[key][word]
                    for key in list(detail_tmp.keys()):
                        if not detail_tmp[key]:
                            del detail_tmp[key]
                    if detail_tmp:
                        for key in detail_tmp.keys():
                            if len(detail_tmp[key].keys()) >= factor_length:
                                res = self.small_fun(detail, key, detail_ll, res)
        return res
    
    def sovle_res(self, detail, res, factor_length):
        detail_keys = detail.keys()
        if '非敏感' not in detail_keys:
            detail_ll = []
            for key in detail_keys:
                if len(detail[key].keys()) >= factor_length:
                    res = self.small_fun(detail, key, detail_ll, res)
        return res
        
    

    def ac_results(self, query, sentence, use_rule_one=True, use_rule_two=True):
        ner_results = []
        match = []
        category = []
        res = dict()
        for i in self.actree.iter(sentence):
            ner_results.append((i[1][1], i[0] + 1 - len(i[1][1]), i[0] + 1))
        for name in ner_results:
            match.append(name[0])
            category.append(self.dcit[name[0]])
        detail = dict()
        for name in Counter(category).most_common():
            key = name[0]
            detail[key] = dict(Counter([match[i] for i, x in enumerate(category) if x == key]).most_common())
        
        res['judge'] = '非敏感'
        if detail:
            if use_rule_two: #使用规则二
                factor_length = self.factor(query, match) #求解文本的长度因子
            else:
                factor_length = 1 #不使用规则二

            res = self.sovle_res(detail, res, factor_length)
                
        #print(res)
        if use_rule_one:#使用规则一
            res = self.keyword_in_word_seg(sentence, res)
        else:
            res = self.keyword_in_word_seg_old(sentence, res)
        return res
    
    
    def factor(self,content, match):#
        len_keywords = 0
        for word in match:
            len_keywords += len(word)
        len_sentence = len(content)
        #print()
        if len_sentence/len_keywords < 50:
            return 1
        elif len_sentence/len_keywords > 100:
            return 3
        else:
            return 2
    
    #考虑到前一个函数存在漏报的可能，故仅对于两个字以下的词汇和英文词汇使用该方法
    def keyword_in_word_seg(self, sentence, result):
        if result['judge'] == '敏感':
            #仅对于两个字以下的词汇和英文词汇使用该方法
            matched_keywords = [result['detail'][0]['description'][i]['keyword'] for i in range(len(result['detail'][0]['description']))]
            determin = 0
            #print(result)
            for word in matched_keywords:
                #print(word.isalpha())
                if (len(word) < 2) or (word.encode('utf-8').isalpha()==True) or word.isdigit()==True:
                    determin += 1
            if determin > 0:
                count = 0
                word_seg_list = jieba.lcut(sentence) #返回分词后的list #这种方式缺点：jieba无法正确分词的敏感词无法检测出来，从而导致漏报
                #print(word_seg_list)
                for word in matched_keywords:
                    if word in word_seg_list:
                        count += 1
                if count > 0:
                    return result
                else:
                    return {'judge': '非敏感'}
            else:
                return result
        else:
            return {'judge': '非敏感'}
    
#从txt文件中提取出所有的关键词，并保存至 target_sensitive.xlsx 文件中，只在敏感词列表更改时运行一遍
#def write_keywords2excel(keyword_txt, keyword_file_excel=None, update_keyword_file=True):
def obtain_keywords_dic(keyword_txt):
    read_keywords = Txt_Op(keyword_txt)
    keyword_list = read_keywords.read_txt() #读取关键词列表（以回车符分割）
    keyword_dic = read_keywords.list2dic(keyword_list) #将关键词库转换为集合列表（以加号为分隔符切分为两个集合，每个集合中以分号为分隔符分割成不同的关键词）
    return keyword_dic

def obtain_keywords_dic_2(keyword_txt):
    read_keywords = Txt_Op(keyword_txt)
    keyword_list = read_keywords.read_txt() #读取关键词列表（以回车符分割）
    keyword_dic = read_keywords.list2dic_2(keyword_list) #将关键词库转换为集合列表（以加号为分隔符切分为两个集合，每个集合中以分号为分隔符分割成不同的关键词）
    return keyword_dic

#修改下面的程序
def grop_detection(ahocorasick_ner, query, keyword_dic_list, stopwords_list):
    groupDectect = True #是否进行多关键词组合检测
    result_all_sen_type = {'judge':'敏感', 'detail':[], 'type':[]}
    len_content = len(query)
    if len_content > 100:#对于长文本进行分句
        sentence_list = re.split('[：；。！？:;!?\n]', query.strip()) #没有加入英文句号，即点号
    else:#对于短文本，不进行分句
        sentence_list = re.split('[\n]', query.strip())
    
    for num, keyword_dic in enumerate(keyword_dic_list): #对于每一种敏感类型
        if num == 0:#检测是否包含正面关键词
            result = ahocorasick_ner[num].ac_results(query, query, use_rule_one=True, use_rule_two=False)
            #print(result)
            if result != {'judge': '非敏感'}:
                matched_keywords = [result['detail'][0]['description'][i]['keyword'] for i in range(len(result['detail'][0]['description']))]
                matched_result = sets_operation_2(matched_keywords, keyword_dic) #判断检测出的关键词是否包含在 关键词集合列表中
                if matched_result:
                    return {'judge':'非敏感'}
        else:
            c_sentence = 0
            detail_num = dict()
            detail_num_list = []
            type_list = []
            for sentence in sentence_list:#对于文本中的每一个句子
                c_sentence+=1 #文本中句子的数量
                #针对每一个句子进行检测，只要有一个句子完成组合匹配，则认为该文本为敏感文本
                result_sentence = ahocorasick_ner[num].ac_results(query, sentence) #每个句子的敏感关键词信息
                #print(result_sentence)
                if result_sentence != {'judge': '非敏感'}: #如果待检测文本中包含敏感关键词
                    new_sentence = ahocorasick_ner[num].del_stopwords(stopwords_list, sentence)
                    result_sentence = ahocorasick_ner[num].ac_results(query, new_sentence)
                    #print(result_sentence)
                if result_sentence != {'judge': '非敏感'}: #如果待检测文本中包含敏感关键词
                    detail_num = dict()
                    judge = result_sentence['judge']
                    if judge == '敏感': #如果去除停用词后依然（或无需去除停用词）为敏感文档
                        if groupDectect: #若为True，则使用组合关键词；否则使用单关键词
                            matched_keywords = [result_sentence['detail'][0]['description'][i]['keyword'] for i in range(len(result_sentence['detail'][0]['description']))]
                            matched_result = sets_operation_3(len_content, matched_keywords, keyword_dic) #判断检测出的关键词是否包含在 关键词集合列表中(True or False)
                            if matched_result:
                                detail_num = result_sentence['detail'][0] #detail_num为某类型的关键词信息{'description':[{'count':2,'keyword':'哈哈'},{...}],'type':'历史虚无主义'}
                                if detail_num not in detail_num_list:
                                    if detail_num['type'] not in type_list:#前面没有检测到该敏感类型
                                        type_list.append(detail_num['type'])
                                        detail_num_list.append(detail_num) #将每一个句子中 所有满足匹配规则的关键词组合成列表 (去重)
                                    else:#前面已检测到该敏感类型
                                        pass
            if len(detail_num_list) > 0: #将不同敏感类型的关键词组合起来
                for num, item in enumerate(detail_num_list):
                    result_all_sen_type['detail'].append(item)
                    result_all_sen_type['type'].append(type_list[num])
    if result_all_sen_type['detail'] != []:
        return result_all_sen_type
    else:
        return {'judge':'非敏感'}

def create_obj(user_dict_path, custom_sensitive_words):
    keyword_txt_list = ['data/zhengmian.txt','data/shezheng.txt','data/baokong.txt','data/shedu.txt','data/shehuang.txt','data/zhapian.txt','data/zongjiao.txt','data/qita.txt']
    sen_type_list = ['正面','涉政','暴恐','涉赌','涉黄','诈骗','宗教','其他']
    
    ahocorasick_ner = []
    keyword_dic_list = []
    stopword_file = 'data/stop_words.txt'
    add_stopwords = Txt_Op(stopword_file)
    stopwords_list = add_stopwords.read_txt() #读取停用词库
    for num, keyword_txt in enumerate(keyword_txt_list):
        keyword_dic = obtain_keywords_dic_2(keyword_txt)
        keyword_dic_list.append(keyword_dic)
        
        sen_type_c = sen_type_list[num]
        ahocorasick_ner.append(sen_type_c)
        ahocorasick_ner[num] = AhocorasickNer_content()
        ahocorasick_ner[num].add_keywords(user_dict_path, sen_type_c, custom_sensitive_words)  # 添加关键词到AC自动机
    return ahocorasick_ner, stopwords_list, keyword_dic_list

if __name__ == "__main__":
    
    
    
    user_dict_path = "data/target_sensitive_content.xlsx"
    custom_sensitive_words = []
    
    ss = time.time()
    
    query = "共匪出千技巧" 
    
    #对于每一类敏感词库，先分别建立一个AC自动机对象，并分别添加对应的关键词
    ahocorasick_ner, stopwords_list, keyword_dic_list = create_obj(user_dict_path, custom_sensitive_words)
    
    result = grop_detection(ahocorasick_ner, query, keyword_dic_list, stopwords_list)
    #print(result)
    
    print("TIME  : {0}ms!".format(round(1000 * (time.time() - ss), 3))) 
    
    
    
    
