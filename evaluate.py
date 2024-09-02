from ragas import evaluate
from datasets import Dataset
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from ragas.metrics._faithfulness import _statements_output_instructions, StatementsAnswers, _faithfulness_output_instructions, StatementFaithfulnessAnswers
from ragas.metrics._answer_relevance import _output_instructions, AnswerRelevanceClassification
from ragas.metrics._context_precision import _verification_output_instructions, ContextPrecisionVerification
import types
import typing as t

# 自定义faithfulness使用的函数, 适应中文场景
def _create_statements_prompt_ch(self, row: t.Dict) -> PromptValue:
    assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

    text, question = row["answer"], row["question"]
    sentences = self.sentence_segmenter.segment(text)
    sentences = [
        sentence for sentence in sentences if sentence.strip().endswith("。")
    ]
    sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
    prompt_value = self.statement_prompt.format(
        question=question, answer=text, sentences=sentences
    )
    return prompt_value

class RagEvaluator:
    def __init__(self):
        # 自定义faithfulness使用的prompt
        long_form_answer_ch = Prompt(
            name="long_form_answer_ch",
            output_format_instruction=_statements_output_instructions,
            instruction="给定一个问题、一个答案和答案中的句子，分析每个句子的复杂度，并将每个句子分解为一个或多个易于理解的陈述句，同时确保每个陈述句中不使用代词。以 JSON 格式输出结果。",
            examples=[
                {
                    "question": "阿尔伯特·爱因斯坦是谁？他最著名的贡献是什么？",
                    "answer": "他是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的物理学家之一。他最著名的贡献是发展了相对论，同时也为量子力学的发展做出了重要贡献。",
                    "sentences": """
                0:他是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的物理学家之一。
                1:他最著名的贡献是发展了相对论，同时也为量子力学的发展做出了重要贡献。
                """,
                    "analysis": StatementsAnswers.parse_obj(
                        [
                            {
                                "sentence_index": 0,
                                "simpler_statements": [
                                    "阿尔伯特·爱因斯坦是一位出生于德国的理论物理学家。",
                                    "阿尔伯特·爱因斯坦被广泛认为是有史以来最伟大和最有影响力的物理学家之一。",
                                ],
                            },
                            {
                                "sentence_index": 1,
                                "simpler_statements": [
                                    "阿尔伯特·爱因斯坦最著名的贡献是发展了相对论。",
                                    "阿尔伯特·爱因斯坦也为量子力学的发展做出了重要贡献。",
                                ],
                            },
                        ]
                    ).dicts(),
                }
            ],
            input_keys=["question", "answer", "sentences"],
            output_key="analysis",
            language="chinese",
        )
        nli_statements_ch = Prompt(
            name="nli_statements_ch",
            instruction="您的任务是根据给定的上下文判断一系列陈述的忠实度。对于每一个陈述，如果能直接从上下文中推断出来，则返回 1；如果不能直接从上下文中推断出来，则返回 0。",
            output_format_instruction=_faithfulness_output_instructions,
            examples=[
                {
                    "context": """张明是XYZ大学的学生。他正在攻读计算机科学学位。本学期他注册了几门课程，包括数据结构、算法和数据库管理。张明是一名勤奋的学生，花费大量时间学习和完成作业。他经常在图书馆工作到很晚来完成项目。""",
                    "statements": [
                        "张明的专业是生物学。",
                        "张明正在上人工智能课程。",
                        "张明是一名专心致志的学生。",
                        "张明有一份兼职工作。",
                    ],
                    "answer": StatementFaithfulnessAnswers.parse_obj(
                        [
                            {
                                "statement": "张明的专业是生物学。",
                                "reason": "张明的专业被明确提到是计算机科学。没有信息表明他主修生物学。",
                                "verdict": 0,
                            },
                            {
                                "statement": "张明正在上人工智能课程。",
                                "reason": "上下文中提到了张明目前注册的课程，但没有提到人工智能课程。因此，无法推断出张明正在上人工智能课程。",
                                "verdict": 0,
                            },
                            {
                                "statement": "张明是一名专心致志的学生。",
                                "reason": "上下文表明他花费大量时间学习和完成作业。此外，他还经常在图书馆工作到很晚来完成项目，这表明他非常专心。",
                                "verdict": 1,
                            },
                            {
                                "statement": "张明有一份兼职工作。",
                                "reason": "上下文中没有提及张明有一份兼职工作。",
                                "verdict": 0,
                            },
                        ]
                    ).dicts(),
                },
                {
                    "context": """光合作用是一种由植物、藻类和某些细菌用来将光能转化为化学能的过程。""",
                    "statements": ["爱因斯坦是一位天才。"],
                    "answer": StatementFaithfulnessAnswers.parse_obj(
                        [
                            {
                                "statement": "爱因斯坦是一位天才。",
                                "reason": "上下文和陈述无关。",
                                "verdict": 0,
                            }
                        ]
                    ).dicts(),
                },
            ],
            input_keys=["context", "statements"],
            output_key="answer",
            output_type="json",
            language="chinese"
        )
        faithfulness.statement_prompt = long_form_answer_ch
        faithfulness.nli_statements_message = nli_statements_ch
        faithfulness._create_statements_prompt = types.MethodType(_create_statements_prompt_ch, faithfulness) 
        self.faithfulness = faithfulness

        # 自定义answer_relevancy使用的prompt
        question_generation_ch = Prompt(
            name="question_generation_ch",
            instruction="""为给定的答案生成一个问题，并判断答案是否含糊其辞。如果答案含糊其辞，则将非承诺性标记为 1；如果答案明确，则标记为 0。含糊其辞的答案是指回避、模糊或不确定的回答。例如，“我不知道”或“我不确定”就是含糊其辞的答案""",
            output_format_instruction=_output_instructions,
            examples=[
                {
                    "answer": """爱因斯坦出生于德国。""",
                    "context": """爱因斯坦是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。""",
                    "output": AnswerRelevanceClassification.parse_obj(
                        {
                            "question": "爱因斯坦出生在哪里？",
                            "noncommittal": 0,
                        }
                    ).dict(),
                },
                {
                    "answer": """它可以根据周围环境的温度改变皮肤颜色。""",
                    "context": """最近的一项科学研究发现，在亚马逊雨林中有一种新物种的青蛙，它具有根据周围环境温度改变皮肤颜色的独特能力。""",
                    "output": AnswerRelevanceClassification.parse_obj(
                        {
                            "question": "新发现的青蛙物种有什么独特的能力？",
                            "noncommittal": 0,
                        }
                    ).dict(),
                },
                {
                    "answer": """珠穆朗玛峰""",
                    "context": """地球上最高的山峰，从海平面测量，是一座位于喜马拉雅山脉的著名山峰。""",
                    "output": AnswerRelevanceClassification.parse_obj(
                        {
                            "question": "地球上最高的山峰是什么？",
                            "noncommittal": 0,
                        }
                    ).dict(),
                },
                {
                    "answer": """我不知道2023年发明的智能手机的突破性功能，因为我对2022年之后的信息不了解。""",
                    "context": """2023年宣布了一项突破性的发明：一款电池续航一个月的智能手机，彻底改变了人们使用移动技术的方式。""",
                    "output": AnswerRelevanceClassification.parse_obj(
                        {
                            "question": "2023年发明的智能手机的突破性功能是什么？",
                            "noncommittal": 1,
                        }
                    ).dict(),
                },
            ],
            input_keys=["answer", "context"],
            output_key="output",
            output_type="json",
            language="chinese"
        )
        answer_relevancy.question_generation = question_generation_ch
        self.answer_relevancy = answer_relevancy

        # 自定义context_utilization使用的prompt
        context_precision_cn = Prompt(
            name="context_precision_cn",
            instruction="""给定问题、答案和上下文，验证上下文对于得出给定答案是否有用。有用则输出 "1"，无用则输出 "0"，以 JSON 格式呈现结果.""",
            output_format_instruction=_verification_output_instructions,
            examples=[
                {
                    "question": """你能告诉我关于阿尔伯特·爱因斯坦的什么信息？""",
                    "context": """阿尔伯特·爱因斯坦（1879年3月14日—1955年4月18日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他最著名的贡献是发展了相对论，同时在量子力学方面也做出了重要贡献，是20世纪初现代物理学革命重塑自然科学理解的核心人物之一。他的质能等价公式 E = mc² 被誉为“世界上最著名的方程”。他在1921年获得诺贝尔物理学奖，以表彰他在理论物理学方面的贡献，特别是他发现了光电效应定律，这是量子理论发展的关键一步。他的工作也因其对科学哲学的影响而闻名。1999年，英国《物理世界》杂志对全球130位顶尖物理学家进行了投票，爱因斯坦被评为有史以来最伟大的物理学家。他的智力成就和原创性使爱因斯坦成为天才的代名词。""",
                    "answer": """阿尔伯特·爱因斯坦生于1879年3月14日，是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他在1921年获得了诺贝尔物理学奖，以表彰他在理论物理学方面的贡献。1905年，他发表了4篇论文。爱因斯坦于1895年移居瑞士。""",
                    "verification": ContextPrecisionVerification(
                        reason="提供的上下文确实有助于得出给定的答案。上下文中包含了关于阿尔伯特·爱因斯坦生活和贡献的关键信息，这些信息在答案中都有所体现。",
                        verdict=1,
                    ).dict(),
                },
                {
                    "question": """谁赢得了2020年国际板球理事会（ICC）世界杯？""",
                    "context": """2022年国际板球理事会（ICC）男子T20世界杯于2022年10月16日至11月13日在澳大利亚举行，这是第八届该赛事。原计划于2020年举办，但因COVID-19疫情而推迟。英格兰队在决赛中以5个小门击败巴基斯坦队，赢得他们的第二个ICC男子T20世界杯冠军。""",
                    "answer": """英格兰队。""",
                    "verification": ContextPrecisionVerification(
                        reason="上下文有用，因为它澄清了关于2020年ICC世界杯的情况，并指出英格兰队是实际上在2022年举行的、原定于2020年的赛事的胜者。",
                        verdict=1,
                    ).dict(),
                },
                {
                    "question": """世界上最高的山峰是什么？""",
                    "context": """安第斯山脉是世界上最长的大陆山脉，位于南美洲。它跨越七个不同的国家，拥有西半球许多最高山峰。这个山脉以其多样的生态系统著称，包括高原的安第斯高原和亚马逊雨林。""",
                    "answer": """珠穆朗玛峰。""",
                    "verification": ContextPrecisionVerification(
                        reason="提供的上下文讨论的是安第斯山脉，虽然壮观，但并未包含珠穆朗玛峰的相关信息，也没有直接关联到关于世界上最高山峰的问题。",
                        verdict=0,
                    ).dict(),
                },
            ],
            input_keys=["question", "context", "answer"],
            output_key="verification",
            output_type="json",
            language="chinese"
        )
        context_utilization.context_precision_prompt = context_precision_cn
        self.context_utilization = context_utilization

    def evaluate(self, llm, embedding, rag_results):
        dataset = Dataset.from_dict(rag_results)
        result = evaluate(dataset, metrics=[self.faithfulness, self.answer_relevancy, self.context_utilization], llm = llm, embeddings = embedding)
        return result