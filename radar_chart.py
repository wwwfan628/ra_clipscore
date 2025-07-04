import matplotlib.pyplot as plt
import numpy as np

# 示例数据
labels = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 5, 2, 4]

# 设置角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 闭合数据
values += values[:1]
angles += angles[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制雷达图
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, alpha=0.25)

# 添加标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 设置范围
ax.set_ylim(0, 5)


values2 = [2, 4, 4, 3, 5]
values2 += values2[:1]
ax.plot(angles, values2, linestyle='dashed', color='orange')
ax.fill(angles, values2, alpha=0.25, color='orange')


plt.title("Radar Chart Example")
plt.show()


