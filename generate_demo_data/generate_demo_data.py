import pandas as pd
import numpy as np
import os


def generate_demo_csv():
    # 1. 提供的 58 个精确 AIM-SNPs 列表
    snps_str = "rs10008492 rs10497191 rs1063 rs10741584 rs11034709 rs11104947 rs11224765 rs11652805 rs116783706 rs11841589 rs12006467 rs12130799 rs12229892 rs12498138 rs1250233 rs1288367 rs1371048 rs149768401 rs17034666 rs17207681 rs17451739 rs174570 rs17599827 rs17822931 rs2024566 rs2042762 rs2261033 rs239031 rs2642066 rs2717329 rs2851012 rs28777 rs2920295 rs3217805 rs3809161 rs434124 rs464165 rs4704322 rs4907251 rs530357165 rs534029120 rs535319466 rs546642722 rs56142203 rs570435573 rs59950930 rs6078837 rs6123723 rs6548616 rs6969817 rs722869 rs7554936 rs7596027 rs7745461 rs7799912 rs79200067 rs9285110 rs9522149"
    snp_columns = snps_str.split()

    if len(snp_columns) != 58:
        print(f"警告: 预期有 58 个 SNPs，但实际解析到 {len(snp_columns)} 个。")

    # 2. 模拟的基因型分布
    genotypes = [11, 21, 22]

    # 固定随机种子，以保证任何人运行生成的数据都是一致的
    np.random.seed(42)

    # 3. 生成 100 个 SNH (陕南汉) 和 100 个 ZHG (壮族) 样本
    print("正在生成 SNH 和 ZHG 群体的脱敏虚拟分型数据...")
    snh_samples = [[f"SNH_demo{i}"] + list(np.random.choice(genotypes, size=len(snp_columns))) for i in range(1, 101)]
    zhg_samples = [[f"ZHG_demo{i}"] + list(np.random.choice(genotypes, size=len(snp_columns))) for i in range(1, 101)]

    data = snh_samples + zhg_samples
    columns = ["Group"] + snp_columns

    df_demo = pd.DataFrame(data, columns=columns)

    # 4. 确保目标输出文件夹存在并保存文件
    os.makedirs("data", exist_ok=True)
    output_path = "data/sample_demo.csv"

    df_demo.to_csv(output_path, index=False)



if __name__ == "__main__":
    generate_demo_csv()