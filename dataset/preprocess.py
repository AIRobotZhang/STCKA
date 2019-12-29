# -*- coding: utf-8 -*-

import tagme
import logging
import sys
import os.path
import requests
import json
from tqdm import tqdm

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "d866f962-a8f3-4213-a93b-fc0c1383a973-843339462"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

def get_instance_concept(file):
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            if ent not in ent_concept:
                ent_concept[ent] = []
            ent_concept[ent].append(cpt)

    return ent_concept

def Annotation_mentions(txt):
    """
    发现那些文本中可以是维基概念实体的概念
    :param txt: 一段文本对象，str类型
    :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
    """
    annotation_mentions = tagme.mentions(txt)
    dic = dict()
    for mention in annotation_mentions.mentions:
        try:
            dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
        except:
            logger.error('error annotation_mention about ' + mention)
    return dic


def Annotate(txt, language="en", theta=0.1):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(A, B):score]  A为文本当中的概念实体，B为维基概念实体，score为其得分
    """
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    try:
        for ann in annotations.get_annotations(theta):
            # print(ann)
            try:
                A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                dic[(A, B)] = score
            except:
                logger.error('error annotation about ' + ann)
    except:
        pass
    return dic


if __name__ == '__main__':
    import spacy
    import time
    file = 'F:/downloads/data-concept/data-concept/data-concept-instance-relations.txt'
    k = 5
    nlp = spacy.load("en_core_web_sm")
    ent_concept = get_instance_concept(file)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
    f_w = open('tagmynews.tsv', 'w', encoding='utf-8')
    text = 'None'
    with open('tagmynews.txt', 'r', encoding='utf-8') as f:
        for ii, line in enumerate(f):
            if ii % 8 == 6:
                label = line.strip()
                f_w.write(text+'\t'+' '.join(concept)+'\t'+label+'\n')
            if ii % 8 != 0:
                continue
            line = line.strip()
            text = line
            # text = ' '.join(line[:-1])
            # label = line[-1]
            # txt = 'Jay and Jolin are born in Taiwan'
            # obj = Annotate(text, theta=0.1)

            doc = nlp(text)
            obj = []

            for ent in doc.ents:
                obj.append(ent.text)
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            concept = []
            if len(obj) == 0:
                concept.append('None')

            for ent in obj:
                # concept_sen = "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance="+i+"&topK=3"
                # response = requests.get(concept_sen, headers=headers)

                # content = json.loads(response.content)
                if ent in ent_concept:
                    length = len(ent_concept[ent])
                    length = k if length > k else length
                    concept.extend(ent_concept[ent][0:length])
                else:
                    concept.append(ent)


                # if len(content) > 0:
                #     # print(content.keys())
                #     concept.extend(content.keys())
                # else:
                #     concept.append(i)
                # time.sleep(0.5)
            # print(concept)
            # print(' '.join(concept))
            
    f_w.close()

