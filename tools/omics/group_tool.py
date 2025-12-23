import pandas as pd
import os

class BioDataGroupTool:
    def __init__(self, df):
        self.df = df
        # 常用药物列表
        self.drugs = ['Lenvatinib', 'Sorafenib', 'Regorafenib', 'Apatinib']

    def get_groups(self, rule_type, param=None):
        """
        主入口函数
        :param rule_type: 分组规则类型 (gender, age, patient_drug, organoid_drug, mutation, expression)
        :param param: 辅助参数 (比如具体的药物名、基因名、年龄阈值)
        :return: (Group1_DF, Group2_DF, Group1_Name, Group2_Name)
        """
        rule_type = rule_type.lower()

        # 性别分组
        if rule_type in ['gender', '性别']:
            return self._group_by_str('Gender', 'male', 'female')
        # 年龄分组
        elif rule_type in ['age', '年龄']:
            if isinstance(param, int):
                mid = param
            else:
                mid = self.df['Age'].median()
            return self.__group_by_num('Age', mid)
        # 患者用药分组
        elif rule_type in ['patient_drug', '患者药物']:
            if not param:
                raise ValueError("请提供药物名称作为参数,例如 'Lenvatinib'")
            col_name = f"Patient {param}"
            return self._group_by_str(col_name, 'Resistant', 'Sensitive')
        # 类器官用药分组
        elif rule_type in ['organoid_drug', '类器官药物']:
            if not param:
                raise ValueError("请提供药物名称作为参数,例如 'Lenvatinib'")
            col_name = f"Organoid-{param}-Sensitive"
            return self._group_by_str(col_name, 'Yes', 'No', label1='Resistant', label2='Sensitive')
        # 突变基因分组
        elif rule_type in ['mutation', '突变']:
            if not param:
                raise ValueError("请提供基因名称作为参数,例如 'TP53'")
            if param.startswith('GENE_'):
                col_name = param
            else:
                col_name = f"GENE_{param}"
            return self._group_by_str(col_name, 1, 0, label1='Mutated', label2='Wild-Type')
        # 基因表达分组
        elif rule_type in ['expression', '表达量']:
            if not param:
                raise ValueError("请提供基因名称作为参数,例如 'TP53'")
            if param.startswith('RNA_'):
                col_name = param
            else:
                col_name = f"RNA_{param}"
            if col_name not in self.df.columns:
                print(f"基因表达列 {col_name} 不存在于数据中。")
                return None, None, None, None
            mid = self.df[col_name].median()
            return self.__group_by_num(col_name, mid, label_mode='high_low')
        else:
            print(f"未知的分组规则类型: {rule_type}")
            return None, None, None, None
        
    def _group_by_str(self, col_name, val1, val2, label1=None, label2=None):
        """
        按字符串值分组
        :param col_name: 列名
        :param val1: 分组1的值
        :param val2: 分组2的值
        :param label1: 分组1的标签
        :param label2: 分组2的标签
        :return: (Group1_DF, Group2_DF, Group1_Name, Group2_Name)
        """
        if col_name not in self.df.columns:
            print(f"❌ 列不存在: {col_name}")
            return None, None, None, None
        
        group1_df = self.df[self.df[col_name] == val1]
        group2_df = self.df[self.df[col_name] == val2]
        
        group1_name = label1 if label1 else str(val1)
        group2_name = label2 if label2 else str(val2)
        
        print(f"分组依据: {col_name}, {group1_name} 样本数: {len(group1_df)}, {group2_name} 样本数: {len(group2_df)}")
        return group1_df, group2_df, group1_name, group2_name
    
    def __group_by_num(self, col_name, mid, label_mode='high_low'):
        """
        按数值中位数分组
        :param col_name: 列名
        :param mid: 中位数阈值
        :param label_mode: 标签模式
        :return: (Group1_DF, Group2_DF, Group1_Name, Group2_Name)
        """
        if col_name not in self.df.columns:
            print(f"❌ 列不存在: {col_name}")
            return None, None, None, None
        
        group1_df = self.df[self.df[col_name] > mid]
        group2_df = self.df[self.df[col_name] <= mid]

        if label_mode == 'high_low':
            group1_name = f"High ({col_name}>={mid})"
            group2_name = f"Low ({col_name}<{mid})"
        else:
            group1_name = f">= {mid}"
            group2_name = f"< {mid}"
        
        print(f"分组成功: {col_name}: {group1_name} 样本数: {len(group1_df)}, {group2_name} 样本数: {len(group2_df)}")
        return group1_df, group2_df, group1_name, group2_name
    
# 示例用法
if __name__ == "__main__":
    filepath = "D:/Bit/tools/data/最终三表合一数据.csv"
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
    df = pd.read_csv(filepath)
    tool = BioDataGroupTool(df)
    print("\n--- 分组测试 ---")
    g1, g2, n1, n2 = tool.get_groups("类器官药物", "Lenvatinib")

    if g1 is not None:
        print(f"Group 1 Name: {n1}, Sample Count: {len(g1)}")
        print(f"Group 2 Name: {n2}, Sample Count: {len(g2)}")

    out_dir = "D:/Bit/tools/data/group_output"
    path1 = f"{out_dir}/group1.csv"
    path2 = f"{out_dir}/group2.csv"

    g1.to_csv(path1, index=False, encoding='utf-8-sig')
    g2.to_csv(path2, index=False, encoding='utf-8-sig')
