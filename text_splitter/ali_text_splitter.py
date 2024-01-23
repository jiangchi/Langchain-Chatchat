from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            """ 作用：将三个或更多连续的换行符（\n）替换为一个换行符。
            注释：去除多余的空行，将多个空行缩减为一个。 """
            text = re.sub(r"\n{3,}", r"\n", text)
            """ 作用：将所有空白字符（包括空格、制表符、换行符等）替换为一个空格。
            注释：去除文本中的多余空白，将多个连续的空白字符合并为一个空格。 """
            text = re.sub('\s', " ", text)
            """ 作用：将连续的两个换行符（\n\n）替换为空字符串。
            注释：去除文本中的双重换行，将连续的两个换行符删除。 """
            text = re.sub("\n\n", "", text)
        try:
            from modelscope.pipelines import pipeline
        except ImportError:
            raise ImportError(
                "Could not ipackagemport modelscope python . "
                "Please install modelscope with `pip install modelscope`. "
            )


        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
