import pandas as pd
import json
import os
import glob
import argparse
import numpy as np

def clean_alibaba_traces(input_dir, output_file, target_dag_count=20000):
    print(f"🔍 开始扫描输入目录: {input_dir}")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"📄 找到 {len(csv_files)} 个 CSV 文件待处理。")
    
    dags_list = []
    collected_count = 0
    
    # 阿里 Trace 提取出来的核心字段（根据官方 README 定义）
    # 通常 CallGraph 包含：timestamp, traceid, rpcid, um, rpctype, interface, dm, rt
    usecols = ['traceid', 'um', 'dm', 'rt']
    
    for file in csv_files:
        if collected_count >= target_dag_count:
            break
            
        print(f"⏳ 正在处理文件: {os.path.basename(file)} | 已收集 DAG 数量: {collected_count}/{target_dag_count}")
        
        try:
            # 💡 核心防 OOM 技巧：使用 chunksize 分块读取巨大的 CSV
            chunk_iter = pd.read_csv(file, usecols=usecols, chunksize=100000, on_bad_lines='skip', engine='c')
            
            for chunk in chunk_iter:
                # 过滤掉无效的微服务名称 (如 NaN, '(?)', '')
                chunk = chunk.dropna(subset=['um', 'dm', 'traceid'])
                chunk = chunk[~chunk['um'].isin(['(?)', 'UNKNOWN', ''])]
                chunk = chunk[~chunk['dm'].isin(['(?)', 'UNKNOWN', ''])]
                
                # 按 traceid 分组提取 DAG
                grouped = chunk.groupby('traceid')
                
                for trace_id, group in grouped:
                    nodes_dict = {}
                    edges_list = []
                    
                    for _, row in group.iterrows():
                        u, v = str(row['um']), str(row['dm'])
                        
                        # RT (Response Time) 可能包含正负值（代表不同视角的记录），取绝对值作为 workload 预估
                        # 转换为空值容错，如果 RT 解析失败则赋一个基础值
                        try:
                            rt_val = abs(float(row['rt']))
                            # 过滤掉极端异常的耗时 (比如大于 5000ms 的可能是严重 timeout)
                            rt_val = np.clip(rt_val, 1.0, 5000.0) 
                        except:
                            rt_val = 10.0 # 默认补底值
                            
                        # 记录节点 workload (如果一个节点被调用多次，这里取最大耗时或累加均可，此处取最大值)
                        if u not in nodes_dict:
                            nodes_dict[u] = {"workload": 5.0} # 上游节点给一个默认极小值
                        if v not in nodes_dict:
                            nodes_dict[v] = {"workload": rt_val}
                        else:
                            nodes_dict[v]["workload"] = max(nodes_dict[v]["workload"], rt_val)
                            
                        edges_list.append([u, v])
                    
                    # 💡 拓扑质量控制：只要节点数在 5 到 30 之间的中大型 DAG，太小没意义，太大 GNN 难以收敛
                    num_nodes = len(nodes_dict)
                    if 5 <= num_nodes <= 30:
                        dags_list.append({
                            "trace_id": str(trace_id),
                            "nodes": nodes_dict,
                            "edges": edges_list
                        })
                        collected_count += 1
                        
                        if collected_count >= target_dag_count:
                            break
                            
                if collected_count >= target_dag_count:
                    break
                    
        except Exception as e:
            print(f"❌ 读取文件 {file} 时发生错误: {e}")
            continue

    print(f"\n✅ 数据提取完毕！共提取高质量 DAG 数量: {len(dags_list)}")
    
    # 保存为最终的 JSON 供 RL 环境读取
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dags_list, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON 文件已保存至: {output_file} (文件大小约: {os.path.getsize(output_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Microservice Trace Cleaner")
    parser.add_argument("--input_dir", type=str, required=True, help="存放解压后 CSV 的目录")
    parser.add_argument("--output_file", type=str, default="alibaba_dags.json", help="输出的 JSON 路径")
    parser.add_argument("--target_count", type=int, default=20000, help="需要提取的 DAG 数量")
    args = parser.parse_args()
    
    clean_alibaba_traces(args.input_dir, args.output_file, args.target_count)