import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image, params):
    # 中值滤波降噪
    median = cv2.medianBlur(image, params['median_kernel'])
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    
    # 背景分离（白色背景）
    lower_white = np.array([0, 0, params['bg_threshold']])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    foreground = cv2.bitwise_and(median, median, mask=~mask)
    
    # 提取V通道进行增强
    v_channel = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], 
                          tileGridSize=(params['clahe_tile'], params['clahe_tile']))
    enhanced_v = clahe.apply(v_channel)
    
    # 合并增强后的通道
    hsv[:, :, 2] = enhanced_v
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 边缘检测（Canny算法）
    edges = cv2.Canny(enhanced_img, 
                     params['canny_low'], 
                     params['canny_high'])
    
    # 形态学处理连接边缘
    kernel = np.ones((params['morph_kernel'], params['morph_kernel']), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, 
                                  kernel, iterations=params['morph_iter'])
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # 特征分析
    defect_contours = []
    color_features = []
    shape_features = []
    
    avg_hue = np.mean(hsv[:,:,0][closed_edges == 255])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params['min_area']:
            continue
        
        # 形状特征（基于最小外接圆）
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = area / (np.pi * radius**2) if radius != 0 else 0
        
        # 颜色特征（基于边缘区域）
        mask = np.zeros_like(closed_edges)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_hue = cv2.mean(hsv[:,:,0], mask=mask)[0]
        hue_diff = abs(mean_hue - avg_hue)
        
        # 分类逻辑
        if circularity < params['circ_threshold'] or hue_diff > params['hue_threshold']:
            defect_contours.append(cnt)
        
        color_features.append(hue_diff)
        shape_features.append(circularity)
    
    return {
        'median': median,
        'hsv': hsv,
        'enhanced': enhanced_img,
        'edges': edges,
        'closed_edges': closed_edges,
        'contours': contours,
        'defects': defect_contours,
        'avg_hue': avg_hue,
        'color_features': color_features,
        'shape_features': shape_features
    }

# Streamlit界面
st.title("🌾糙米不完善粒检测系统")


# 侧边栏参数设置
with st.sidebar:
    st.header("参数设置")
    
    # 预处理参数
    st.subheader("预处理")
    params = {
        'median_kernel': st.selectbox("中值滤波核大小", [3, 5, 7], index=1),
        'bg_threshold': st.slider("背景分离阈值", 150, 230, 200),
        'clahe_clip': st.slider("CLAHE对比度限制", 1.0, 5.0, 3.0),
        'clahe_tile': st.selectbox("CLAHE网格大小", [4, 8, 16], index=1)
    }
    
    # 边缘检测参数
    st.subheader("边缘检测")
    params.update({
        'canny_low': st.slider("Canny低阈值", 30, 100, 50, key="canny_low"),
        'canny_high': st.slider("Canny高阈值", 100, 200, 150, key="canny_high"),
        'morph_kernel': st.selectbox("形态学核大小", [3, 5, 7], index=0, key="morph_kernel"),
        'morph_iter': st.slider("形态学迭代次数", 1, 5, 2, key="morph_iter")
    })
    
    # 特征分析参数
    st.subheader("特征分析")
    params.update({
        'min_area': st.number_input("最小面积阈值", 50, 300, 100, key="min_area"),
        'circ_threshold': st.slider("圆形度阈值", 0.3, 0.9, 0.65, 0.05, key="circ_threshold"),
        'hue_threshold': st.slider("色调差异阈值", 10, 30, 20, key="hue_threshold")
    })
    
    st.markdown("---")
    st.caption("提示：调整参数后会自动重新计算")

# 主界面
uploaded_file = st.file_uploader("上传糙米图像", 
                                type=["jpg", "png", "jpeg"],
                                key="main_uploader")

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = process_image(bgr_img, params)
    
    # 显示处理流程
    st.subheader("图像处理流程")
    cols = st.columns(3)
    cols[0].image(cv2.cvtColor(results['median'], cv2.COLOR_BGR2RGB), 
                  caption="中值滤波降噪", use_container_width=True)
    cols[1].image(results['hsv'][:,:,0], 
                  caption="HSV-H通道", clamp=True, use_container_width=True)
    cols[2].image(cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB), 
                  caption="CLAHE增强", use_container_width=True)

    cols = st.columns(2)
    cols[0].image(results['edges'], 
                  caption=f"Canny边缘检测 ({params['canny_low']}-{params['canny_high']})", 
                  clamp=True, use_container_width=True)
    cols[1].image(results['closed_edges'], 
                  caption=f"闭合边缘 (核{params['morph_kernel']}x{params['morph_kernel']})", 
                  clamp=True, use_container_width=True)

    # 显示检测结果
    result_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(result_img, results['defects'], -1, (255,0,0), 2)

    st.subheader(f"检测结果：发现{len(results['defects'])}个不完善粒")
    st.image(result_img, use_container_width=True)

    # 统计信息
    st.markdown("### 特征分布")
    col1, col2 = st.columns(2)
    col1.bar_chart(results.get('color_features', []), color="#ffaa0088")
    col1.caption(f"颜色差异分布 (阈值={params['hue_threshold']})")
    col2.bar_chart(results.get('shape_features', []), color="#00ffaa88")
    col2.caption(f"形状特征分布 (阈值={params['circ_threshold']:.2f})")

    # 添加下载按钮
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    st.download_button(
        label="下载检测报告",
        data=result_pil.tobytes(),
        file_name="检测结果.png",
        mime="image/png"
    )
else:
    st.info("请上传包含糙米的图像文件")

st.markdown("---")
st.caption(f"当前检测标准：圆形度<{params['circ_threshold']} 或 色调差异>{params['hue_threshold']}")
st.caption("Developed by LSQ | 检测算法版本3.0")